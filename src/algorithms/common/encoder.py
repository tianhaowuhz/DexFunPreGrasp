from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from tasks.isaacgym_utils import pack_pointcloud_observations

from .network import MLP, get_activation


class TupleIndexer(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Forward pass of the TupleIndexer module.

        Args:
            x (Tuple[torch.Tensor, ...]): Input tuple of tensors.

        Returns:
            torch.Tensor: The tensor at the specified index.
        """
        return x[self.index]


class ObservationEncoder(nn.Module):
    num_tactile_channels: int
    num_gradient_channels: int
    num_state_channels: int
    num_full_state_channels: int

    num_pointcloud_points: int

    out_channels: int

    def __init__(self, observation_info: List[Dict[str, Any]], pointcloud_mask: bool = True):
        super().__init__()
        self.observation_info = observation_info

        self.pointcloud_mask = pointcloud_mask

        self.num_pointcloud_points: int = 0
        self.num_tactile_channels: int = 0
        self.num_gradient_channels: int = 0
        self.num_state_channels: int = 0
        self.num_full_state_channels: int = 0

        for info in observation_info:
            if "tactile" in info["tags"]:
                self.num_tactile_channels += info["dim"]
            elif "gradient" in info["tags"]:
                self.num_gradient_channels += info["dim"]
            elif "pointcloud" in info["tags"]:
                self.num_pointcloud_points += info["dim"] // 3
            else:
                self.num_state_channels += info["dim"]
        self.num_full_state_channels = self.num_state_channels + self.num_tactile_channels + self.num_gradient_channels

    def parse_observations(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse the observations into named_observations.

        Args:
            observations (torch.Tensor): The observations tensor. 2D or 3D tensor.
                - Shape: (batch_size, num_observations) or (batch_size, seq_len, num_observations)

        Returns:
            Dict[str, torch.Tensor]: The named_observations, the dictionary contains the following keys:
                - "pointcloud": (batch_size, [seq_len,] in_point_channels, num_pointcloud_points)
                - "tactile": (batch_size, [seq_len,] num_tactile_channels)
                - "gradient": (batch_size, [seq_len,] num_gradient_channels)
                - "state": (batch_size, [seq_len,] num_state_channels)
                - "full_state": (batch_size, [seq_len,] num_full_state_channels)
        """
        assert observations.ndim == 2 or observations.ndim == 3, "Observations must be 2D or 3D tensor"

        device = observations.device
        batch_size = observations.size(0)
        seq_len = observations.size(1) if observations.ndim == 3 else 1

        # initialize the named_observations
        named_observations = {}
        if self.num_pointcloud_points > 0:
            named_observations["pointcloud"] = {}
        if self.num_tactile_channels > 0:
            named_observations["tactile"] = []
        if self.num_gradient_channels > 0:
            named_observations["gradient"] = []
        named_observations["state"] = []

        # parse the observations
        for info in self.observation_info:
            if "tactile" in info["tags"]:
                tactile = observations[..., info["start"] : info["end"]]
                named_observations["tactile"].append(tactile)
            elif "gradient" in info["tags"]:
                gradient = observations[..., info["start"] : info["end"]]
                named_observations["gradient"].append(gradient)
            elif "pointcloud" in info["tags"]:
                pointcloud = observations[..., info["start"] : info["end"]]
                if observations.ndim == 2:
                    pointcloud = pointcloud.reshape(batch_size, -1, 3)
                elif observations.ndim == 3:
                    pointcloud = pointcloud.reshape(batch_size * seq_len, -1, 3)
                named_observations["pointcloud"][info["name"]] = {}
                named_observations["pointcloud"][info["name"]]["points"] = pointcloud
            else:
                state = observations[..., info["start"] : info["end"]]
                named_observations["state"].append(state)

        # concatenate the observations
        named_observations["state"] = torch.cat(named_observations["state"], dim=-1).to(device)
        if self.num_tactile_channels > 0:
            named_observations["tactile"] = torch.cat(named_observations["tactile"], dim=-1).to(device)
        if self.num_gradient_channels > 0:
            named_observations["gradient"] = torch.cat(named_observations["gradient"], dim=-1).to(device)
        if self.num_pointcloud_points > 0:
            named_observations["pointcloud"] = pack_pointcloud_observations(
                named_observations["pointcloud"], mask=self.pointcloud_mask, device=device
            )

        # create full state
        full_state = [named_observations["state"]]
        if self.num_tactile_channels > 0:
            full_state.append(named_observations["tactile"])
        if self.num_gradient_channels > 0:
            full_state.append(named_observations["gradient"])
        named_observations["full_state"] = torch.cat(full_state, dim=-1).to(device)

        if "pointcloud" in named_observations and observations.ndim == 3:
            named_observations["pointcloud"] = named_observations["pointcloud"].reshape(
                batch_size, seq_len, -1, self.num_pointcloud_points
            )

        return named_observations


class DummyObservationEncoder(ObservationEncoder):
    def __init__(self, observation_info: List[Dict[str, Any]]):
        super().__init__(observation_info=observation_info)
        self.out_channels = self.num_full_state_channels

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.parse_observations(observations)["full_state"]


class MultiPathObservationEncoder(ObservationEncoder):
    def __init__(
        self,
        observation_info: List[Dict[str, Any]],
        embedding_channels: int,
        hidden_channels: Optional[List[int]] = None,
        activation_name: str = "relu",
        point_feature_extractor: Optional[nn.Module] = None,
        out_point_channels: int = 512,
        pointcloud_mask: bool = True,
    ):
        super().__init__(observation_info=observation_info, pointcloud_mask=pointcloud_mask)
        assert self.num_pointcloud_points == 0 or point_feature_extractor is not None
        self.out_channels = embedding_channels

        hidden_channels = [] if hidden_channels is None else hidden_channels
        activation = get_activation(activation_name)
        num_paths = 1

        self.state_encoder = MLP(
            in_channels=self.num_state_channels,
            out_channels=embedding_channels,
            hidden_channels=hidden_channels,
            activation=activation,
        )
        if self.num_tactile_channels > 0:
            num_paths += 1
            self.tactile_encoder = MLP(
                in_channels=self.num_tactile_channels,
                out_channels=embedding_channels,
                activation=activation,
            )
        if self.num_gradient_channels > 0:
            num_paths += 1
            self.gradient_encoder = MLP(
                in_channels=self.num_gradient_channels,
                out_channels=embedding_channels,
                activation=activation,
            )
        if self.num_pointcloud_points > 0:
            num_paths += 1
            self.pointcloud_encoder = nn.Sequential(
                OrderedDict(
                    [
                        ("feature_extractor", point_feature_extractor),
                        ("indexer", TupleIndexer(index=0)),
                        (
                            "encoder",
                            MLP(
                                in_channels=out_point_channels,
                                out_channels=embedding_channels,
                                activation=activation,
                            ),
                        ),
                    ]
                )
            )

        if num_paths > 1:
            self.fuse = MLP(
                in_channels=embedding_channels * num_paths,
                out_channels=embedding_channels,
                activation=activation,
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        named_observations = self.parse_observations(observations)
        state_embedding = self.state_encoder(named_observations["state"])
        embeddings = [state_embedding]

        if self.num_tactile_channels > 0:
            tactile_embedding = self.tactile_encoder(named_observations["tactile"])
            embeddings.append(tactile_embedding)
        if self.num_gradient_channels > 0:
            gradient_embedding = self.gradient_encoder(named_observations["gradient"])
            embeddings.append(gradient_embedding)
        if self.num_pointcloud_points > 0:
            pointcloud_embedding = self.pointcloud_encoder(named_observations["pointcloud"])
            embeddings.append(pointcloud_embedding)

        if len(embeddings) > 1:
            embeddings = self.fuse(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]

        return embeddings


class PointCloudBasedObservationEncoder(ObservationEncoder):
    """Refactor this from `src.algorithms.DDIM.policy.DDIM`"""

    state_encoder: nn.Module
    point_encoder: nn.Module

    def __init__(
        self,
        observation_info: List[Dict[str, Any]],
        point_feature_extractor: nn.Module,
        embedding_channels: int = 512,
        hidden_channels: int = 512,
        out_point_channels: int = 512,
        pointcloud_mask: bool = True,
    ):
        super().__init__(observation_info=observation_info, pointcloud_mask=pointcloud_mask)
        self.state_encoder = MLP(
            in_channels=self.num_full_state_channels,
            out_channels=embedding_channels,
            hidden_channels=[hidden_channels],
            activation=nn.ReLU(),
        )
        self.point_encoder = nn.Sequential(
            OrderedDict(
                [
                    ("feature_extractor", point_feature_extractor),
                    ("indexer", TupleIndexer(index=0)),
                    (
                        "encoder",
                        MLP(
                            in_channels=out_point_channels,
                            out_channels=embedding_channels,
                            hidden_channels=[hidden_channels],
                            activation=nn.ReLU(),
                        ),
                    ),
                ]
            )
        )
        self.out_channels = embedding_channels * 2

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        named_observations = self.parse_observations(observations)

        states = named_observations["full_state"]
        pointclouds = named_observations["pointcloud"]

        is_sequence: bool = states.ndim == 3

        if is_sequence:
            batch_size, seq_len, _ = states.size()
            states = states.view(-1, states.size(-1))
            pointclouds = pointclouds.view(-1, pointclouds.size(-2), pointclouds.size(-1))

        state_embeddings = self.state_encoder(states)
        pointcloud_embeddings = self.point_encoder(pointclouds)
        embeddings = torch.cat([pointcloud_embeddings, state_embeddings], dim=-1)

        if is_sequence:
            embeddings = embeddings.view(batch_size, seq_len, -1)

        return embeddings

    def freeze_pointcloud_encoder(self) -> None:
        for param in self.point_encoder.parameters():
            param.requires_grad = False


class SinglePathObservationEncoder(ObservationEncoder):
    """Refactor this from `src.algorithms.DDIM.policy.DDIM`"""

    state_encoder: nn.Module

    def __init__(
        self,
        observation_info: List[Dict[str, Any]],
        embedding_channels: int = 512,
        hidden_channels: Union[int, List[int]] = 512,
        activation_name: str = "relu",
    ):
        super().__init__(observation_info=observation_info)
        hidden_channels = [hidden_channels] if isinstance(hidden_channels, int) else hidden_channels
        activation: nn.Module = get_activation(activation_name)
        self.state_encoder = MLP(
            in_channels=self.num_full_state_channels,
            out_channels=embedding_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            activation_last=True,
        )
        self.out_channels = embedding_channels

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        named_observations = self.parse_observations(observations)
        assert "pointcloud" not in named_observations, "Pointcloud is not supported in SinglePathObservationEncoder"

        states = named_observations["full_state"]

        is_sequence: bool = states.ndim == 3

        if is_sequence:
            batch_size, seq_len, _ = states.size()
            states = states.view(-1, states.size(-1))

        embeddings = self.state_encoder(states)

        if is_sequence:
            embeddings = embeddings.view(batch_size, seq_len, -1)

        return embeddings


class DaggerConcatSinglePathObservationEncoder(ObservationEncoder):
    """Refactor this from `src.algorithms.DDIM.policy.DDIM`"""

    state_encoder: nn.Module

    def __init__(
        self,
        observation_info: List[Dict[str, Any]],
        embedding_channels: int = 512,
        hidden_channels: Union[int, List[int]] = 512,
        activation_name: str = "relu",
    ):
        super().__init__(observation_info=observation_info)
        hidden_channels = [hidden_channels] if isinstance(hidden_channels, int) else hidden_channels
        activation: nn.Module = get_activation(activation_name)
        self.state_encoder = MLP(
            in_channels=self.num_full_state_channels * 2,  # for append: self.num_full_state_channels*2
            out_channels=embedding_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            activation_last=True,
        )
        self.out_channels = embedding_channels

    # for append
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        named_observations = self.parse_observations(observations)
        assert "pointcloud" not in named_observations, "Pointcloud is not supported in SinglePathObservationEncoder"

        states = named_observations["full_state"]

        is_sequence: bool = states.ndim == 3

        if is_sequence:
            batch_size, seq_len, _ = states.size()
            states = states.view(-1, seq_len * states.size(-1))

        embeddings = self.state_encoder(states)

        return embeddings
