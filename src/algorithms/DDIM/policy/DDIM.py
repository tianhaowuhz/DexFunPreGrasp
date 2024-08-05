import math
import time
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import rearrange, reduce

from algorithms.common.network import MLP
from algorithms.DDIM.model.common.normalizer import LinearNormalizer
from algorithms.DDIM.model.diffusion.mask_generator import LowdimMaskGenerator
from algorithms.DDIM.model.diffusion.positional_embedding import SinusoidalPosEmb
from algorithms.DDIM.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from algorithms.DDIM.policy.base_pc_policy import BasePointcloudPolicy

# TODO: fix this hack
FT_INDICES = [0, 3, 6, 9, 12]
PCL = False

class DDIM(BasePointcloudPolicy):
    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        # task params
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        # arch
        n_layer=8,
        n_cond_layers=0,
        n_head=4,
        p_drop_emb=0.0,
        p_drop_attn=0.3,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
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
        args=None,
        use_normalizer=False,  # TODO no normalizer for now
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        self.use_normalizer = use_normalizer
        self.space = space
        self.action_type = action_type
        self.args = args

        if space == "riemann":
            # TODO configure from action info, currently set to 27, because 3 are xyz, no need to riemann
            action_dim *= 2
        # create conditional encoder

        self.obs_tactile_dim = obs_tactile_dim
        print(self.obs_tactile_dim)
        # pointcloud
        num_points = pcl_number

        self.full_obs_dim = obs_state_dim + num_points * 3

        self.encode_state_type = encode_state_type
        if encode_state_type == "all":
            encode_state_dim = obs_state_dim
        else:
            encode_state_dim = 0
            # magic number
            if "arm" in encode_state_type:
                encode_state_dim += 7
            if "dof" in encode_state_type:
                encode_state_dim += 30
            if "fingertippose" in encode_state_type:
                encode_state_dim += 35
            if "obj2palmpose" in encode_state_type:
                encode_state_dim += 7
            if "target" in encode_state_type:
                encode_state_dim += 25
            if "bbox" in encode_state_type:
                encode_state_dim += 6
            if "tactile" in encode_state_type:
                if "fttactile" in encode_state_type:
                    encode_state_dim += 5
                else:
                    encode_state_dim += self.obs_tactile_dim

        if args is not None and args.cond_on_arm and action_type == "hand":
            encode_state_dim += arm_action_dim

        print("Diffusion Model:")
        print(f"  - encode_state_dim: {encode_state_dim}")
        print(f"  - num_points: {num_points}")

        # state encoder
        self.state_enc = nn.Sequential(
            nn.Linear(encode_state_dim, hidden_dim),
            nn.ReLU(False),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(False),
        )

        if PCL:
            # obj pcl feature encoder
            self.obj_enc = PointNetEncoder(
                global_feat=True, out_dim=num_points, feature_transform=False, channel=3
            )  # for pointnet

            self.obj_global_enc = nn.Sequential(
                nn.Linear(num_points, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
                nn.ReLU(),
            )

        # create diffusion model
        if PCL:
            obs_feature_dim = embed_dim * 2
        else:
            obs_feature_dim = embed_dim

        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            # n_emb=cond_dim, # for pcl
            n_emb=1024,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= model  ============
    def obs_encoder(self, batch_size, nobs_pc, nobs_state, obs_steps):
        # encoder pc obs
        # reshape B, T, ... to B*T
        if PCL:
            this_nobs_pc = nobs_pc[:, :obs_steps, ...].reshape(-1, *nobs_pc.shape[2:])
            nobs_features, _, _ = self.obj_enc(this_nobs_pc)
            nobs_features = self.obj_global_enc(nobs_features)
            # reshape back to B, T, Do
            cond_pc = nobs_features.reshape(batch_size, obs_steps, -1)

        # encoder state obs TODO different with SDE
        # reshape B, T, ... to B*T
        this_nobs_state = nobs_state[:, :obs_steps, ...].reshape(-1, *nobs_state.shape[2:])
        nobs_features = self.state_enc(this_nobs_state)
        # reshape back to B, T, Do
        cond_state = nobs_features.reshape(batch_size, obs_steps, -1)

        if PCL:
            cond_state = torch.cat([cond_pc, cond_state], dim=-1)
        return cond_state

    def action2grad(self, x, inv=False, action_type=None):
        if not inv:
            assert action_type is not None

            if action_type == "arm" or action_type == "hand" or action_type == "all":
                batch_size = x.size(0)
                horizon = x.size(1)
                state_dim = x.size(2)
                x = torch.cat(
                    [
                        torch.sin(x).reshape(batch_size, horizon, state_dim, 1),
                        torch.cos(x).reshape(batch_size, horizon, state_dim, 1),
                    ],
                    -1,
                ).reshape(batch_size, horizon, state_dim * 2)

            return x
        else:
            if len(x.size()) == 4:
                step = x.size(0)
                batch_size = x.size(1)
                horizon = x.size(2)
                state_dim = x.size(3)

                if action_type == "arm" or action_type == "hand" or action_type == "all":
                    x = x.reshape(step, batch_size, horizon, int(state_dim / 2), 2)
                    x = torch.atan2(x[:, :, :, :, 0:1], x[:, :, :, :, 1:2]).reshape(
                        step, batch_size, horizon, int(state_dim / 2)
                    )

                return x
            elif len(x.size()) == 3:
                batch_size = x.size(0)
                horizon = x.size(1)
                state_dim = x.size(2)

                if action_type == "arm" or action_type == "hand" or action_type == "all":
                    x = x.reshape(batch_size, horizon, int(state_dim / 2), 2)
                    x = torch.atan2(x[:, :, :, 0:1], x[:, :, :, 1:2]).reshape(batch_size, horizon, int(state_dim / 2))

                return x

    def process_batch(self, batches):
        action_batch, obj_batch, state_batch = batches
        processed_batch = dict()
        if self.space == "riemann":
            action_batch = self.action2grad(action_batch, action_type=self.action_type)

        if self.encode_state_type == "all":
            encode_state_batch = state_batch
        else:
            encode_state_batch = torch.tensor([]).to(state_batch.device)
            # magic number
            if "arm" in self.encode_state_type:
                encode_state_batch = torch.cat([encode_state_batch, state_batch[:, :, :7]], dim=-1)
            if "dof" in self.encode_state_type:
                encode_state_batch = torch.cat([encode_state_batch, state_batch[:, :, 7:37]], dim=-1)
            if "fingertippose" in self.encode_state_type:
                encode_state_batch = torch.cat([encode_state_batch, state_batch[:, :, 67:102]], dim=-1)
            if "obj2palmpose" in self.encode_state_type:
                encode_state_batch = torch.cat([encode_state_batch, state_batch[:, :, 132:139]], dim=-1)
            if "target" in self.encode_state_type:
                encode_state_batch = torch.cat([encode_state_batch, state_batch[:, :, 152:177]], dim=-1)
            if "bbox" in self.encode_state_type:
                encode_state_batch = torch.cat([encode_state_batch, state_batch[:, :, 202:208]], dim=-1)
            if "tactile" in self.encode_state_type:
                if "fttactile" in self.encode_state_type:
                    tactile_batch = state_batch[:, :, 208 : 208 + self.obs_tactile_dim]
                    tactile_batch = tactile_batch[:, :, FT_INDICES]
                    encode_state_batch = torch.cat([encode_state_batch, tactile_batch], dim=-1)
                else:
                    encode_state_batch = torch.cat(
                        [encode_state_batch, state_batch[:, :, 208 : 208 + self.obs_tactile_dim]], dim=-1
                    )
            if self.args is not None and self.args.cond_on_arm and self.action_type == "hand":
                encode_state_batch = torch.cat([encode_state_batch, state_batch[:, :, -6:]], dim=-1)

        processed_batch["action"] = action_batch
        processed_batch["pc"] = obj_batch
        processed_batch["state"] = encode_state_batch
        return processed_batch

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        cond=None,
        generator=None,
        random_traj=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        if random_traj is not None:
            trajectory = random_traj
        else:
            trajectory = torch.randn(
                size=condition_data.shape, dtype=condition_data.dtype, device=condition_data.device, generator=generator
            )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, batch, random_traj=True):
        # normalize input
        batch = self.process_batch(batch)
        if self.use_normalizer:
            nobs_pc = self.normalizer.normalize(batch["pc"])
            nobs_state = self.normalizer.normalize(batch["state"])
            nactions = self.normalizer["action"].normalize(batch["action"])
        else:
            nobs_pc = batch["pc"]
            nobs_state = batch["state"]
            nactions = batch["action"]

        B = nactions.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = self.obs_encoder(B, nobs_pc, nobs_state, To)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            nobs_features = self.obs_encoder(B, nobs_pc, nobs_state, T)
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # st = time.time()
        # run sampling
        if not random_traj:
            random_action = nactions
        else:
            random_action = None
        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond, random_traj=random_action, **self.kwargs)
        # print(time.time() - st)
        # unnormalize prediction
        if self.use_normalizer:
            naction_pred = nsample[..., :Da]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)
        else:
            action_pred = nsample[..., :Da]

        # get action
        # if self.pred_action_steps_only:
        #     action = action_pred
        # else:
        #     start = To - 1
        #     end = start + self.n_action_steps
        #     action = action_pred[:,start:end]

        # result = {
        #     'action': action,
        #     'action_pred': action_pred
        # }
        # return result

        if self.space == "riemann":
            action_pred = self.action2grad(action_pred, inv=True, action_type=self.action_type)
        return action_pred

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append({"params": self.state_enc.parameters(), "weight_decay": obs_encoder_weight_decay})
        if PCL:
            optim_groups.append({"params": self.obj_enc.parameters(), "weight_decay": obs_encoder_weight_decay})
            optim_groups.append({"params": self.obj_global_enc.parameters(), "weight_decay": obs_encoder_weight_decay})
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        batch = self.process_batch(batch)

        if self.use_normalizer:
            nobs_pc = self.normalizer.normalize(batch["pc"])
            nobs_state = self.normalizer.normalize(batch["state"])
            nactions = self.normalizer["action"].normalize(batch["action"])
        else:
            nobs_pc = batch["pc"]
            nobs_state = batch["state"]
            nactions = batch["action"]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            cond = self.obs_encoder(batch_size, nobs_pc, nobs_state, To)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:, start:end]
        else:
            nobs_features = self.obs_encoder(batch_size, nobs_pc, nobs_state, horizon)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

    # TODO: move this part out of DDIM
    def compute_supervised_loss(self, batch):
        # normalize input
        batch = self.process_batch(batch)
        if self.use_normalizer:
            nobs_pc = self.normalizer.normalize(batch["pc"])
            nobs_state = self.normalizer.normalize(batch["state"])
            nactions = self.normalizer["action"].normalize(batch["action"])
        else:
            nobs_pc = batch["pc"]
            nobs_state = batch["state"]
            nactions = batch["action"]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            cond = self.obs_encoder(batch_size, nobs_pc, nobs_state, To)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:, start:end]
        else:
            nobs_features = self.obs_encoder(batch_size, nobs_pc, nobs_state, horizon)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        zeros = torch.zeros(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]

        timesteps = torch.zeros(bsz, device=trajectory.device).long()

        pred = self.model(zeros, timesteps, cond)
        target = trajectory

        loss_mask = ~condition_mask

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

    def predict_supervised(self, batch):
        # normalize input
        batch = self.process_batch(batch)
        if self.use_normalizer:
            nobs_pc = self.normalizer.normalize(batch["pc"])
            nobs_state = self.normalizer.normalize(batch["state"])
            nactions = self.normalizer["action"].normalize(batch["action"])
        else:
            nobs_pc = batch["pc"]
            nobs_state = batch["state"]
            nactions = batch["action"]

        B = nactions.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = self.obs_encoder(B, nobs_pc, nobs_state, To)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            nobs_features = self.obs_encoder(B, nobs_pc, nobs_state, T)
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        zeros = torch.zeros(cond_data.shape, device=cond_data.device)
        timesteps = torch.zeros(B, device=cond_data.device).long()

        nsample = self.model(zeros, timesteps, cond)

        # unnormalize prediction
        if self.use_normalizer:
            naction_pred = nsample[..., :Da]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)
        else:
            action_pred = nsample[..., :Da]
        return action_pred


class MLPForDiffusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()
        n_emb = n_emb // 2
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)

        in_channels = (n_obs_steps + horizon + 1) * n_emb
        out_channels = horizon * input_dim
        self.mlp = MLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=[2048, 1024, 512],
            activation_last=False,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size: int = sample.size(0)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps)
        input_emb = self.input_emb(sample)
        cond_obs_emb = self.cond_obs_emb(cond)

        output_shape = sample.shape
        embeddings = torch.cat(
            [cond_obs_emb.reshape(batch_size, -1), input_emb.reshape(batch_size, -1), time_emb], dim=-1
        )
        output = self.mlp(embeddings).reshape(output_shape)
        return output

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
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

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups
