from typing import Any, Dict

import torch
import torch.nn as nn

from algorithms.DDIM.model.diffusion.transformer_for_diffusion import TransformerForDiffusion


class ObservationEncoder(nn.Module):
    num_embedding_channels: int

    def __init__(self, num_embedding_channels: int):
        super().__init__()
        self.num_embedding_channels = num_embedding_channels

    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass


class DexterousTransformer(nn.Module):
    def __init__(
        self,
        out_channels: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 0.00001,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.d_model = d_model

        # consider 1 timestep token and 2 observation tokens
        memory_mask = torch.tril(torch.ones((4, 3), dtype=torch.bool), diagonal=1)
        memory_mask = torch.where(memory_mask, 0.0, float("-inf"))
        self.memory_mask = nn.Parameter(memory_mask, requires_grad=False)

        tgt_mask = torch.tril(torch.ones((4, 4), dtype=torch.bool), diagonal=0)
        tgt_mask = torch.where(tgt_mask, 0.0, float("-inf"))
        self.tgt_mask = nn.Parameter(tgt_mask, requires_grad=False)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )

        self.head = nn.Linear(d_model, out_channels)

    def forward(self, observation_embedding: torch.Tensor) -> torch.Tensor:
        batch_size: int = observation_embedding.size(0)
        device = observation_embedding.device

        zeros = torch.zeros((batch_size, 4, self.d_model), dtype=torch.float32, device=device)
        x = self.transformer.forward(observation_embedding, zeros, memory_mask=self.memory_mask, tgt_mask=self.tgt_mask)

        x = self.head(x)
        return x


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        observation_encoder: ObservationEncoder,
        # task params
        horizon,
        n_action_steps,
        n_obs_steps,
        # architecture params
        n_layer=8,
        n_cond_layers=0,
        n_head=4,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
        pred_action_steps_only=False,
        action_type=None,
        encode_state_type="all",
        space="euler",
        action_dim=0,
        arm_action_dim=0,
        obs_state_dim=0,
        obs_tactile_dim=0,
        pcl_number=0,
        hidden_dim=1024,
        embed_dim=512,
    ):
        super().__init__()
        self.observation_encoder = observation_encoder

        num_embedding_channels = observation_encoder.num_embedding_channels
        num_actions = action_dim

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon

        self.num_actions = num_actions

        self.transformer = TransformerForDiffusion(
            input_dim=num_actions,
            output_dim=num_actions,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=num_embedding_channels,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=num_embedding_channels,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=False,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers,
        )

    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        observation_embedding: torch.Tensor = self.observation_encoder(observation)

        batch_size: int = observation_embedding.size(0)
        device = observation_embedding.device

        zeros = torch.zeros_like((batch_size, self.horizon, self.num_actions), dtype=torch.float32, device=device)
        timesteps = torch.zeros(batch_size, self.horizon, dtype=torch.long, device=device)
        actions = self.transformer(zeros, timesteps, observation_embedding)
        actions = actions[:, self.n_obs_steps - 1, :]

        return actions
