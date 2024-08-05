import os
from copy import deepcopy
from typing import Optional

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from hydra.utils import instantiate
from isaacgymenvs.tasks.base.vec_task import VecTask
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithms.DDIM.model.common.lr_scheduler import get_scheduler
from algorithms.DDIM.model.diffusion.ema_model import EMAModel

from ..algo import Algorithm
from .policy.DDIM import DDIM


class DiffusionPolicy(Algorithm):
    def __init__(
        self,
        env: VecTask,
        cfg_train: DictConfig,
        experiment_name: str = "run",
        device: torch.device = "cpu",
        checkpoint_path: Optional[os.PathLike] = None,
        training: bool = True,
        print_freq: int = 1,
        num_observation_steps: int = 1,
        num_evaluation_rounds: int = 10,
    ):
        super().__init__(env, experiment_name, device, print_freq, num_observation_steps, num_evaluation_rounds)

        self.cfg_train = cfg_train
        self.training = training

        noise_scheduler = DDPMScheduler(**cfg_train["policy"]["scheduler"])
        self.score = DDIM(noise_scheduler=noise_scheduler, **cfg_train["policy"]["network"])
        self.score.to(self.device)

        if checkpoint_path is not None:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.score.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        self.n_obs_steps = self.score.n_obs_steps
        self.horizon = self.score.horizon
        self.action_dim = self.score.action_dim
        self.obs_state_dim = cfg_train["policy"]["network"]["obs_state_dim"]
        self.pcl_number = cfg_train["policy"]["network"]["pcl_number"]
        self.obs_pcl_dim = self.pcl_number * 3
        self.n_action_steps = cfg_train["policy"]["network"]["n_action_steps"]

        # create a counter for multi-step inference
        self.counter = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self.cached_actions = torch.zeros((self.num_envs, self.n_action_steps, self.action_dim), device=self.device)

        observation_space = self.cfg_train["policy"]["observation_space"]
        diffusion_observation_info = self.get_observation_subset(observation_space)

        self.diffusion_observation_info = diffusion_observation_info

        if self.training:
            dataset = instantiate(cfg_train["learn"]["dataset"])
            dataloader = instantiate(cfg_train["learn"]["dataloader"], dataset=dataset)

            self.dataloader: DataLoader = dataloader

            self.num_epochs = self.cfg_train["learn"]["num_epochs"]
            num_training_steps = self.num_epochs * len(self.dataloader)

            self.optimizer = self.score.get_optimizer(**self.cfg_train["policy"]["optimizer"])
            self.ema = EMAModel(model=deepcopy(self.score), **self.cfg_train["policy"]["ema"])
            self.lr_scheduler = get_scheduler(
                optimizer=self.optimizer,
                num_training_steps=num_training_steps,
                last_epoch=-1,
                **self.cfg_train["policy"]["lr_scheduler"],
            )

    def run(self):
        if self.training:
            print(f"Current score model: {self.score.training}")
            self.env.train()
            self.score.train()
            current_iteration = 0
            for epoch in range(self.num_epochs):
                for i, (observations, actions) in enumerate(tqdm(self.dataloader)):
                    current_iteration += 1
                    batch_size: int = observations.shape[0]
                    observations: torch.Tensor = observations.to(self.device)
                    actions: torch.Tensor = actions.to(self.device)

                    states = observations[:, :, : self.obs_state_dim].clone().to(self.device).float()
                    pointclouds = (
                        observations[:, :, self.obs_state_dim : self.obs_state_dim + self.obs_pcl_dim]
                        .clone()
                        .to(self.device)
                        .float()
                        .reshape(batch_size, self.horizon, self.pcl_number, 3)
                    )
                    pointclouds = pointclouds.permute(0, 1, 3, 2)
                    # pointclouds[:, :, :, :self.pcl_number // 2] = pointclouds[:, :, :, self.pcl_number // 2:]
                    # pointclouds.zero_()

                    loss = self.score.compute_loss((actions, pointclouds, states))
                    # loss = self.score.compute_supervised_loss((actions, pointclouds, states))
                    self.writer.add_scalar("Train/loss", loss.item(), current_iteration)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    self.ema.step(self.score)

                if epoch % self.print_freq == 0:
                    self.score.eval()
                    self.env.eval()
                    self.eval(epoch)
                    self.score.train()
                    self.env.train()

                    torch.save(self.score.cpu().state_dict(), self.log_dir / f"score_{epoch}.pt")
                    # torch.save(self.score.obj_enc.cpu().state_dict(), self.log_dir / f"pointnet_{epoch}.pt")

                    self.score.to(self.device)

        else:
            self.score.eval()
            self.eval(0)
            # self.save_robot_trajectories()

    def on_validation_epoch_start(self, iteration: int) -> None:
        self.current_action = torch.randn(self.num_envs, self.horizon, self.action_dim).to(self.device).float()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        actions = torch.zeros(self.num_envs, self.num_actions).to(self.device).float()

        observations = observations.reshape(self.num_envs, self.n_obs_steps, -1).clone()
        states = observations[:, :, : self.obs_state_dim].clone().to(self.device).float()
        pointclouds = (
            observations[:, :, self.obs_state_dim : self.obs_state_dim + self.obs_pcl_dim]
            .clone()
            .to(self.device)
            .float()
            .reshape(self.num_envs, self.n_obs_steps, self.pcl_number, 3)
        )
        pointclouds = pointclouds.permute(0, 1, 3, 2)
        # pointclouds[:, :, :, :self.pcl_number // 2] = pointclouds[:, :, :, self.pcl_number // 2:]
        # pointclouds.zero_()

        # get step actions
        # o2 -> a2, thus n_onb_steps - 1
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps

        if self.num_envs == 1:
            mini_b = 1
        else:
            mini_b = 4

        # optional update `counter` using env.progress_buf (hack)
        self.counter = torch.where(self.env.progress_buf == 0, torch.zeros_like(self.counter), self.counter)
        updated_sign = (self.counter==0)
        total_env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        for i in range(mini_b):
            batch_start = (i * self.num_envs) // mini_b
            batch_end = ((i + 1) * self.num_envs) // mini_b

            res = self.score.predict_action(
                (
                    self.current_action[batch_start:batch_end],
                    pointclouds[batch_start:batch_end],
                    states[batch_start:batch_end],
                ),
                random_traj=False,
            )
            # res = self.score.predict_supervised((self.current_action[batch_start:batch_end], pointclouds[batch_start:batch_end], states[batch_start:batch_end]))
            step_actions = res[:, start:end, :].to(torch.float32)

            # current_action = torch.zeros(self.num_envs, self.horizon, self.action_dim).to(self.device).float()
            # current_action[:, :self.n_obs_steps, :] = grad[:, end-1:end-1+self.n_obs_steps, :].clone()
            self.current_action[batch_start:batch_end, :, :] = step_actions[:, -1:, :].clone()
            
            updated_ids_in_mini = updated_sign[batch_start:batch_end].nonzero(as_tuple=False).squeeze(-1)
            updated_ids_in_all = total_env_ids[batch_start:batch_end][updated_ids_in_mini]
            self.cached_actions[updated_ids_in_all] = step_actions[updated_ids_in_mini].clone()

            actions[batch_start:batch_end] = self.cached_actions[total_env_ids, self.counter][batch_start:batch_end].clone()

        # update counter
        self.counter = (self.counter + 1) % self.n_action_steps

        return actions