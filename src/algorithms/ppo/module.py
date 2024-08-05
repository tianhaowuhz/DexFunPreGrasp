import numpy as np
import torch
import torch.nn as nn
from ipdb import set_trace
from torch.distributions import MultivariateNormal

# from networks.pointnet import PointNetEncoder

local = False


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        model_cfg,
        asymmetric=False,
        pointnet_type="pt2",
        observation_info=None,
        hand_pcl=False,
        hand_model=None,
        args=None,
    ):
        super(ActorCritic, self).__init__()
        # network parameter
        self.asymmetric = asymmetric
        self.pointnet_type = pointnet_type
        """Get network input output dim."""
        # retrival observation dim for input
        self.state_dim = 0
        self.tactile_dim = 0
        self.pcl_dim = 0
        self.grad_dim = 0

        for info in observation_info:
            if "tactile" in info["tags"]:
                self.tactile_dim += info["dim"]
            elif "pointcloud" in info["tags"]:
                self.pcl_dim += info["dim"]
            elif "gradient" in info["tags"]:
                self.grad_dim += info["dim"]
            else:
                self.state_dim += info["dim"]

        print(">>> Initialize ActorCritic")
        print(f"  - state_dim: {self.state_dim}")
        print(f"  - tactile_dim: {self.tactile_dim}")
        print(f"  - pcl_dim: {self.pcl_dim}")
        print(f"  - grad_dim: {self.grad_dim}")
        self.observation_info = observation_info

        # retrival action dim
        self.action_dim = actions_shape[0]
        """
        init network: current we set self.state_base = False, only set true for pure state input
        """
        # network parameter
        activation = get_activation(model_cfg["activation"])
        self.shared_pointnet = model_cfg["shared_pointnet"]
        self.points_per_object = model_cfg["points_per_object"]
        """Actor layer."""
        # state encoder
        actor_state_encoder_hid_sizes = model_cfg["pi_state_encoder_hid_sizes"]
        actor_hidden_dim = actor_state_encoder_hid_sizes[-1]
        self.actor_state_enc = self.build_block(
            self.state_dim, actor_hidden_dim, activation, actor_state_encoder_hid_sizes
        )
        self.total_feat_num = 1

        # tactile feature encoder
        if self.tactile_dim > 0:
            self.actor_tactile_enc = self.build_block(self.tactile_dim, actor_hidden_dim, activation, [])
            self.total_feat_num += 1
        # pointcloud feature encoder
        if self.pcl_dim > 0:
            self.pcl_feature_dim = model_cfg["pcl_feature_dim"]
            self.actor_pcl_enc = self.build_block(self.pcl_feature_dim, actor_hidden_dim, activation, [])
            self.total_feat_num += 1
        # gradient feature encoder
        if self.grad_dim > 0:
            self.actor_grad_enc = self.build_block(self.grad_dim, actor_hidden_dim, activation, [])
            self.total_feat_num += 1

        # fuse feature
        if self.total_feat_num > 1:
            self.actor_fuse = self.build_block(actor_hidden_dim * self.total_feat_num, actor_hidden_dim, activation, [])

        # mlp output
        self.actor_output = self.build_block(
            actor_hidden_dim, self.action_dim, activation, [], activate_for_last_layer=False
        )
        """Critic layer."""
        # state encoder
        critic_state_encoder_hid_sizes = model_cfg["vf_state_encoder_hid_sizes"]
        critic_hidden_dim = critic_state_encoder_hid_sizes[-1]
        self.critic_state_enc = self.build_block(
            self.state_dim, critic_hidden_dim, activation, critic_state_encoder_hid_sizes
        )

        # tactile feature encoder
        if self.tactile_dim > 0:
            self.critic_tactile_enc = self.build_block(self.tactile_dim, critic_hidden_dim, activation, [])

        # pointcloud feature encoder
        if self.pcl_dim > 0:
            self.critic_pcl_enc = self.build_block(self.pcl_feature_dim, critic_hidden_dim, activation, [])

        # gradient feature encoder
        if self.grad_dim > 0:
            self.critic_grad_enc = self.build_block(self.grad_dim, critic_hidden_dim, activation, [])

        # fuse feature
        if self.total_feat_num > 1:
            # mlp output
            self.critic_fuse = self.build_block(
                critic_hidden_dim * self.total_feat_num, critic_hidden_dim, activation, []
            )

        # mlp output
        self.critic_output = self.build_block(critic_hidden_dim, 1, activation, [], activate_for_last_layer=False)

        if self.pcl_dim > 0:
            """Shared layer."""
            if self.shared_pointnet:
                if self.pointnet_type == "pt":
                    self.pointnet_enc = PointNetEncoder(num_points=self.pcl_dim, out_dim=self.pcl_feature_dim)
            else:
                if self.pointnet_type == "pt":
                    self.actor_pointnet_enc = PointNetEncoder(num_points=self.pcl_dim, out_dim=self.pcl_feature_dim)
                    self.critic_pointnet_enc = PointNetEncoder(num_points=self.pcl_dim, out_dim=self.pcl_feature_dim)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def build_block(self, input_dim, output_dim, activation, hidden_dim, activate_for_last_layer=True):
        layers = []
        if len(hidden_dim) == 0:
            layers.append(nn.Linear(input_dim, output_dim))
            if activate_for_last_layer:
                layers.append(activation)
        else:
            layers.append(nn.Linear(input_dim, hidden_dim[0]))
            layers.append(activation)
            for l in range(len(hidden_dim)):
                if l == len(hidden_dim) - 1:
                    layers.append(nn.Linear(hidden_dim[l], output_dim))
                    if activate_for_last_layer:
                        layers.append(activation)
                else:
                    layers.append(nn.Linear(hidden_dim[l], hidden_dim[l + 1]))
                    layers.append(activation)
        return nn.Sequential(*layers)

    def forward(self):
        raise NotImplementedError

    def forward_actor(self, observations):
        """Process observation."""
        batch_size = observations.size(0)

        state_batch, tactile_batch, pcl_batch, gf_batch = self.process_observations(observations=observations)
        """forward."""
        # state encoder
        state_feat = self.actor_state_enc(state_batch)

        # pointcloud encoder
        if self.tactile_dim > 0:
            tactile_feat = self.actor_tactile_enc(tactile_batch)
        if self.pcl_dim > 0:
            if self.shared_pointnet:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, 3)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.pointnet_enc(pcl_batch)
            else:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, 3)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.actor_pointnet_enc(pcl_batch)
            pcl_feat = self.actor_pcl_enc(pcl_feat.reshape(batch_size, -1))  # B x 512
        if self.grad_dim > 0:
            grad_feat = self.actor_grad_enc(gf_batch)

        # fuse
        x = state_feat
        if self.tactile_dim > 0:
            x = torch.cat([x, tactile_feat], -1)
        if self.pcl_dim > 0:
            x = torch.cat([x, pcl_feat], -1)
        if self.grad_dim > 0:
            x = torch.cat([x, grad_feat], -1)
        if self.total_feat_num > 1:
            x = self.actor_fuse(x)

        # output
        x = self.actor_output(x)
        return x

    def forward_critic(self, observations):
        """Process observation."""
        batch_size = observations.size(0)

        state_batch, tactile_batch, pcl_batch, gf_batch = self.process_observations(observations=observations)
        """forward."""
        # state encoder
        state_feat = self.critic_state_enc(state_batch)

        if self.tactile_dim > 0:
            tactile_feat = self.critic_tactile_enc(tactile_batch)
        # point cloud encoder
        if self.pcl_dim > 0:
            if self.shared_pointnet:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, 3)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.pointnet_enc(pcl_batch)
            else:
                if self.pointnet_type == "pt":
                    pcl_batch = pcl_batch.reshape(batch_size, -1, 3)
                    pcl_batch = pcl_batch.permute(0, 2, 1)
                    pcl_feat, _, _ = self.critic_pointnet_enc(pcl_batch)
            pcl_feat = self.critic_pcl_enc(pcl_feat.reshape(batch_size, -1))  # B x 512
        if self.grad_dim > 0:
            grad_feat = self.critic_grad_enc(gf_batch)

        # fuse
        x = state_feat
        if self.tactile_dim > 0:
            x = torch.cat([x, tactile_feat], -1)
        if self.pcl_dim > 0:
            x = torch.cat([x, pcl_feat], -1)
        if self.grad_dim > 0:
            x = torch.cat([x, grad_feat], -1)

        if self.total_feat_num > 1:
            x = self.critic_fuse(x)

        # output
        x = self.critic_output(x)
        return x

    def process_observations(self, observations: torch.Tensor):
        state_batch = []
        tactile_batch = []
        pcl_batch = []
        gf_batch = []
        for info in self.observation_info:
            if "tactile" in info["tags"]:
                tactile_batch.append(observations[:, info["start"] : info["end"]])
            elif "pointcloud" in info["tags"]:
                pcl_batch.append(observations[:, info["start"] : info["end"]])
            elif "gradient" in info["tags"]:
                gf_batch.append(observations[:, info["start"] : info["end"]])
            else:
                state_batch.append(observations[:, info["start"] : info["end"]])

        state_batch = torch.cat(state_batch, dim=-1)

        if self.tactile_dim > 0:
            tactile_batch = torch.cat(tactile_batch, dim=-1)
        else:
            tactile_batch = None

        if self.pcl_dim > 0:
            pcl_batch = torch.cat(pcl_batch, dim=-1)
        else:
            pcl_batch = None

        if self.grad_dim > 0:
            gf_batch = torch.cat(gf_batch, dim=-1)
        else:
            gf_batch = None
        return state_batch, tactile_batch, pcl_batch, gf_batch

    def act(self, observations, states):
        actions_mean = self.forward_actor(observations)

        # print(self.log_std)
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.forward_critic(observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
        )

    def cal_actions_log_prob(self, observations, actions):
        actions_mean = self.forward_actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        return actions.detach(), actions_log_prob.detach(), actions_mean.detach()

    def act_inference(self, observations):
        actions_mean = self.forward_actor(observations)
        return actions_mean

    def evaluate(self, observations, states, actions):
        actions_mean = self.forward_actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.forward_critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


def get_activation(act_name):
    print(act_name)
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        set_trace()
        return None
