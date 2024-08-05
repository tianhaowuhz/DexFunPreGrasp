from typing import List, Optional

import torch
import torch.nn as nn

from .network import MLP, get_activation


class MLPBackbone(MLP):
    num_observation_steps: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_observation_steps: int = 1,
        hidden_channels: Optional[List[int]] = None,
        activation_name: str = "relu",
    ):
        activation: nn.Module = get_activation(activation_name)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            activation=activation,
            activation_last=False,
        )
        self.num_observation_steps = num_observation_steps

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 3:
            observations = observations.reshape(observations.size(0), -1)
        return super().forward(observations)
