from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


def get_activation(name: str) -> nn.Module:
    """Get the (PyTorch) activation module based on the given name.

    Args:
        name (str): The name of the activation module.

    Returns:
        nn.Module: The activation module.
    """
    name = name.lower()
    activations: Dict = torch.nn.modules.activation.__dict__
    activations = {k.lower(): v for k, v in activations.items() if isinstance(v, type) and issubclass(v, nn.Module)}
    assert name in activations, f"Activation '{name}' not found"

    return activations[name]()


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
        activation_last: bool = True,
    ):
        super().__init__()
        channels = (
            [in_channels]
            + (hidden_channels if hidden_channels is not None else [])
            + ([out_channels] if out_channels is not None else [])
        )
        assert len(channels) > 1, "channels must not be empty"

        in_channels, out_channels = channels[0], channels[-1]
        hidden_channels = channels[1:-1]
        activation = activation if activation is not None else nn.ReLU()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.activation = activation

        self.model = nn.Sequential()

        if len(hidden_channels) == 0:
            self.model.add_module("linear", nn.Linear(in_channels, out_channels))
            if activation_last:
                self.model.add_module("activation", activation)
        else:
            self.model.add_module("linear0", nn.Linear(in_channels, hidden_channels[0]))
            self.model.add_module("activation0", activation)

            i = 0
            while i < len(hidden_channels) - 1:
                self.model.add_module(f"linear{i+1}", nn.Linear(hidden_channels[i], hidden_channels[i + 1]))
                self.model.add_module(f"activation{i+1}", activation)
                i += 1

            self.model.add_module(f"linear{i+1}", nn.Linear(hidden_channels[-1], out_channels))
            if activation_last:
                self.model.add_module(f"activation{i+1}", activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Network(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mlp_encoder_partial: Any,
        pointcloud_encoder: Optional[nn.Module] = None,
        head_partial: Any = None,
    ):
        super().__init__()
        self.mlp_encoder = mlp_encoder_partial(in_channels=in_channels)
        self.pointcloud_encoder = pointcloud_encoder
        self.head = head_partial(out_channels=out_channels) if head_partial is not None else None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        states = batch["state"]
        pointclouds = batch.get("pointcloud", None)

        state_embedding = self.mlp_encoder(states)
        if pointclouds is not None:
            pointcloud_embedding = self.pointcloud_encoder(pointclouds)
            embedding = torch.cat([state_embedding, pointcloud_embedding], dim=-1)
        else:
            embedding = state_embedding
        if self.head is not None:
            embedding = self.head(embedding)
        return embedding

    def freeze_pointcloud_encoder(self):
        if self.pointcloud_encoder is not None:
            for param in self.pointcloud_encoder.parameters():
                param.requires_grad = False


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_metainfo: List[Dict],
        action_metainfo: List[Dict],
        actor_partial: Any,
        critic_partial: Any,
        pointcloud_encoder_partial: Any = None,
        initial_std: float = 0.3,
        shared_pointcloud_encoder: bool = False,
        pretrained_pointcloud_encoder: Optional[str] = None,
    ):
        super().__init__()
        self.observation_metainfo = observation_metainfo
        self.action_metainfo = action_metainfo

        self.num_observations = sum([info["dim"] for info in self.observation_metainfo])
        self.num_actions = sum([info["dim"] for info in self.action_metainfo])
        self.num_state_observations = sum(
            [info["dim"] for info in self.observation_metainfo if "pointcloud" not in info["tags"]]
        )
        self.num_pointcloud_observations = sum(
            [info["dim"] for info in self.observation_metainfo if "pointcloud" in info["tags"]]
        )

        if shared_pointcloud_encoder:
            pointcloud_encoder = pointcloud_encoder_partial() if pointcloud_encoder_partial is not None else None
            actor_pointcloud_encoder = pointcloud_encoder
            critic_pointcloud_encoder = pointcloud_encoder
        else:
            actor_pointcloud_encoder = pointcloud_encoder_partial() if pointcloud_encoder_partial is not None else None
            critic_pointcloud_encoder = pointcloud_encoder_partial() if pointcloud_encoder_partial is not None else None

        self.actor = actor_partial(
            in_channels=self.num_observations,
            out_channels=self.num_actions,
            pointcloud_encoder=actor_pointcloud_encoder,
        )
        self.critic = critic_partial(
            in_channels=self.num_observations,
            out_channels=1,
            pointcloud_encoder=critic_pointcloud_encoder,
        )

        self.log_std = nn.Parameter(torch.ones(self.num_actions) * np.log(initial_std))

    def parse_observations(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        named_observations = {}
        for info in self.observation_metainfo:
            named_observations[info["name"]] = observations[:, info["start"] : info["end"]]

        states = []
        pointclouds = []
        for info in self.observation_metainfo:
            if "pointcloud" in info["tags"]:
                pointclouds.append(named_observations[info["name"]])
            else:
                states.append(named_observations[info["name"]])

        batch: Dict[str, torch.Tensor] = {}
        batch["state"] = torch.cat(states, dim=-1)
        if len(pointclouds) > 0:
            batch["pointcloud"] = torch.cat(pointclouds, dim=-1)
        return batch

    def freeze_pointcloud_encoder(self):
        self.actor.freeze_pointcloud_encoder()
        self.critic.freeze_pointcloud_encoder()

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self.parse_observations(observations))

    def forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(self.parse_observations(observations))

    def act(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions_mean = self.forward_actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.forward_critic(observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
        )

    def cal_actions_log_prob(self, observations: torch.Tensor, actions: torch.Tensor):
        actions_mean = self.forward_actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        return actions.detach(), actions_log_prob.detach(), actions_mean.detach()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        actions_mean = self.forward_actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):
        actions_mean = self.forward_actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.forward_critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
