import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_embedding_channels: int):
        """Initializes a SinusoidalPositionalEmbedding module.

        Args:
            num_embedding_channels (int): The number of embedding channels.
        """
        super().__init__()
        self.dim = num_embedding_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SinusoidalPositionalEmbedding module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The positional embedding tensor.
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerBackbone(nn.Module):
    in_channels: int
    out_channels: int

    horizon: int
    num_observation_steps: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        horizon: int,
        num_observation_steps: Optional[int] = None,
        num_layers: int = 8,
        num_heads: int = 4,
        num_embedding_channels: int = 768,
        num_condition_channels: int = 768,
        prob_dropout_embedding: float = 0.1,
        prob_dropout_attention: float = 0.1,
        causal_attention: bool = True,
        timestep_token: bool = True,  # special token for timestep in diffusion
        time_as_condition: bool = True,
        num_condition_layers: int = 0,
        single_step_output: bool = False,
    ):
        super().__init__()
        # compute number of tokens for main trunk and condition encoder
        num_observation_steps = horizon if num_observation_steps is None else num_observation_steps

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_observation_steps = num_observation_steps
        self.horizon = horizon
        self.single_step_output = single_step_output

        self.timestep_token = timestep_token

        if timestep_token and time_as_condition:
            T, T_cond = horizon, 1
        elif timestep_token and not time_as_condition:
            T, T_cond = horizon + 1, 0
        else:
            T, T_cond = horizon, 0

        observation_as_condition = num_condition_channels > 0
        if observation_as_condition:
            T_cond += num_observation_steps

        self.num_tokens = T
        self.num_condition_tokens = T_cond
        self.encoder_only = self.num_condition_tokens == 0  # if there is no condition, then it's just a BERT

        # input embedding stem
        self.input_emb = nn.Linear(in_channels, num_embedding_channels)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, num_embedding_channels))
        self.drop = nn.Dropout(prob_dropout_embedding)

        # cond encoder
        self.time_emb = SinusoidalPositionalEmbedding(num_embedding_channels)
        self.cond_obs_emb = None

        if observation_as_condition:
            self.cond_obs_emb = nn.Linear(num_condition_channels, num_embedding_channels)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        if not self.encoder_only:
            # encoder
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, num_embedding_channels))
            if num_condition_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=num_embedding_channels,
                    nhead=num_heads,
                    dim_feedforward=4 * num_embedding_channels,
                    dropout=prob_dropout_attention,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_condition_layers)
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(num_embedding_channels, 4 * num_embedding_channels),
                    nn.Mish(),
                    nn.Linear(4 * num_embedding_channels, num_embedding_channels),
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=num_embedding_channels,
                nhead=num_heads,
                dim_feedforward=4 * num_embedding_channels,
                dropout=prob_dropout_attention,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # important for stability
            )
            self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        else:
            # encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=num_embedding_channels,
                nhead=num_heads,
                dim_feedforward=4 * num_embedding_channels,
                dropout=prob_dropout_attention,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # attention mask
        if causal_attention:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
            mask = torch.where(mask, float("-inf"), float(0.0))
            self.register_buffer("mask", mask)

            if not self.encoder_only:
                diagonal = 2 if time_as_condition else 1
                S = T_cond
                mask = torch.triu(torch.ones((T, S), dtype=torch.bool), diagonal=diagonal)
                mask = torch.where(mask, float("-inf"), float(0.0))
                self.register_buffer("memory_mask", mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(num_embedding_channels)
        self.head = nn.Linear(num_embedding_channels, out_channels)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_condition = time_as_condition
        self.obs_as_cond = observation_as_condition

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPositionalEmbedding,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerBackbone):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def construct_parameter_groups(self, weight_decay: float = 1e-3) -> List[Dict[str, Any]]:
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return groups

    def forward(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        timesteps: Optional[Union[torch.Tensor, int]] = None,
    ) -> torch.Tensor:
        assert observations.ndim == 3
        assert observations.size(1) == self.num_observation_steps or observations.size(1) == self.horizon
        assert actions is None or (actions.ndim == 3 and actions.size(1) == self.horizon)
        observations = observations[:, : self.num_observation_steps, :]

        B = observations.size(0)
        device = observations.device

        if actions is None:
            actions = torch.zeros((B, self.horizon, self.out_channels), dtype=torch.float32, device=device)

        if timesteps is None:
            timesteps = torch.zeros(B, dtype=torch.long, device=device)
        elif not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            timesteps = timesteps.expand(B)

        timestep_embedding = self.time_emb(timesteps).unsqueeze(1)
        action_embedding = self.input_emb(actions)

        if self.encoder_only:
            assert self.timestep_token
            token_embedding = torch.cat([timestep_embedding, action_embedding], dim=1)
            t = token_embedding.size(1)
            position_embedding = self.pos_emb[:, :t, :]
            x = self.drop(token_embedding + position_embedding)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            # encoder
            if self.obs_as_cond:
                observation_embedding = self.cond_obs_emb(observations)
                cond_embedding = torch.cat([timestep_embedding, observation_embedding], dim=1)
            num_condition_tokens = cond_embedding.size(1)
            position_embedding = self.cond_pos_emb[:, :num_condition_tokens, :]
            x = self.drop(cond_embedding + position_embedding)
            x = self.encoder(x)  # WARNING: no mask for transformer encoder layer (num_condition_layers > 0)
            memory = x
            # decoder
            num_tokens = action_embedding.size(1)
            position_embedding = self.pos_emb[:, :num_tokens, :]
            x = self.drop(action_embedding + position_embedding)
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        x = self.ln_f(x)
        x = self.head(x)

        if self.single_step_output:
            x = x[:, self.num_observation_steps - 1, :]
        return x


if __name__ == "__main__":
    obs_feature_dim = 1024
    input_dim = 26
    output_dim = 26
    cond_dim = 1024

    model = TransformerBackbone(
        in_channels=input_dim,
        out_channels=output_dim,
        horizon=4,
        num_observation_steps=2,
        num_condition_channels=cond_dim,
        num_layers=4,
        num_heads=4,
        num_embedding_channels=cond_dim,
        prob_dropout_embedding=0.0,
        prob_dropout_attention=0.3,
        causal_attention=True,
        time_as_condition=True,
        num_condition_layers=0,
    )

    print("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(model)

    model.eval()

    observations = torch.randn(16, 2, 1024)
    actions = torch.randn(16, 4, 26)
    timesteps = torch.randint(0, 10, (16,))

    print(observations.shape, actions.shape, timesteps.shape)
    out = model.forward_as_backbone(observations, actions, timesteps)
    print(out.shape)

    out = model.forward_as_backbone(observations)
    print(out.shape)
    print(out[0, 1])

    out = model.forward_as_backbone(observations, single_step_output=True)
    print(out.shape)
    print(out[0])

    # model = TransformerBackbone(
    #     in_channels=input_dim,
    #     out_channels=output_dim,
    #     horizon=2,
    #     num_condition_channels=cond_dim,
    #     num_layers=4,
    #     num_heads=4,
    #     num_embedding_channels=cond_dim,
    #     prob_dropout_embedding=0.0,
    #     prob_dropout_attention=0.3,
    #     causal_attention=True,
    #     time_as_condition=False,
    #     num_condition_layers=0,
    # )
