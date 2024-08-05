from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .mlp import MLPBackbone
from .transformer import TransformerBackbone


class Model(nn.Module):
    is_sequence: bool
    num_observation_steps: int

    def __init__(
        self,
        encoder_partial: Any,
        backbone_partial: Any,
        observation_info: List[Dict[str, Any]],
        out_channels: int = 1,
    ):
        super().__init__()
        self.encoder: nn.Module = encoder_partial(observation_info=observation_info)
        self.backbone: nn.Module = backbone_partial(out_channels=out_channels)

        assert isinstance(self.backbone, MLPBackbone) or isinstance(self.backbone, TransformerBackbone)
        self.num_observation_steps = self.backbone.num_observation_steps
        self.is_sequence = not (self.num_observation_steps == 1 and isinstance(self.backbone, MLPBackbone))

    def forward(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        timesteps: Optional[Union[torch.LongTensor, int]] = None,
    ) -> torch.Tensor:
        assert observations.ndim == 3 or observations.ndim == 2
        assert self.is_sequence == (observations.ndim == 3)
        assert (not self.is_sequence) or (observations.size(1) >= self.num_observation_steps)
        if self.is_sequence and observations.size(1) > self.num_observation_steps:
            observations = observations[:, : self.num_observation_steps, :]
        embeddings = self.encoder(observations)

        if isinstance(self.backbone, TransformerBackbone):
            return self.backbone(embeddings, actions, timesteps)
        else:
            return self.backbone(embeddings)

    def construct_parameter_groups(
        self, encoder_weight_decay: float = 0.0, backbone_weight_decay: float = 0.0
    ) -> List[Dict[str, Any]]:
        if isinstance(self.backbone, TransformerBackbone):
            groups = self.backbone.construct_parameter_groups(weight_decay=backbone_weight_decay)
        else:
            groups = [{"params": self.backbone.parameters(), "weight_decay": backbone_weight_decay}]
        groups.append({"params": self.encoder.parameters(), "weight_decay": encoder_weight_decay})
        return groups
