import os
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from hydra.utils import instantiate
from isaacgymenvs.tasks.base.vec_task import VecTask
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from .algo import Algorithm
from .DDIM.model.common.lr_scheduler import get_scheduler


class Diffusion(Algorithm):
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
        max_forward_batch_size: int = 1024,
    ):
        super().__init__(env, experiment_name, device, print_freq, num_observation_steps, num_evaluation_rounds)
        self.max_forward_batch_size = max_forward_batch_size

        self.cfg_train = cfg_train
        self.training = training

        self.num_epochs = cfg_train["learn"].get("num_epochs", 100)

        observation_space = self.cfg_train["policy"]["observation_space"]
        observation_info = self.get_observation_subset(observation_space)

        self.noise_scheduler: DDPMScheduler = instantiate(cfg_train["policy"]["noise_scheduler"])
        self.num_train_timesteps: int = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps: int = self.noise_scheduler.config.num_inference_steps

        self.actor = instantiate(
            self.cfg_train["policy"]["model"],
            observation_info=observation_info,
            out_channels=self.num_actions,
        )
        self.actor.to(self.device)

        if checkpoint_path is not None:
            self.actor.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        if self.training:
            dataset = instantiate(self.cfg_train["learn"]["dataset"])
            dataloader = instantiate(self.cfg_train["learn"]["dataloader"], dataset=dataset)

            self.dataloader: DataLoader = dataloader

            parameter_groups = self.actor.construct_parameter_groups(**self.cfg_train["learn"]["regularization"])
            self.optimizer = instantiate(self.cfg_train["learn"]["optimizer"])(parameter_groups)

            self.num_epochs = self.cfg_train["learn"]["num_epochs"]
            num_training_steps = self.num_epochs * len(self.dataloader)
            self.lr_scheduler = get_scheduler(
                optimizer=self.optimizer,
                num_training_steps=num_training_steps,
                last_epoch=-1,
                **self.cfg_train["learn"]["lr_scheduler"],
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size: int = observations.size(0)
        actions = torch.randn((batch_size, self.horizon, self.num_actions), device=self.device)
        for t in self.noise_scheduler.timesteps:
            output = self.actor(observations, actions, t)
            actions = self.noise_scheduler.step(output, t, actions).prev_sample
        return actions

    def run(self):
        if not self.training:
            self.actor.eval()
            self.eval(0)

        else:
            self.env.train()
            self.actor.train()
            current_iteration = 0
            for epoch in range(self.num_epochs):
                for observations, actions in tqdm(self.dataloader):
                    current_iteration += 1
                    batch_size: int = observations.size(0)
                    observations: torch.Tensor = observations.to(self.device)
                    actions: torch.Tensor = actions.to(self.device)

                    timesteps = torch.randint(
                        0, self.num_train_timesteps, (batch_size,), device=self.device, dtype=torch.long
                    )
                    noise = torch.randn(actions.shape, device=self.device, dtype=actions.dtype)

                    noise_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

                    predictions = self.actor(observations, noise_actions, timesteps)
                    loss = F.mse_loss(predictions, noise)

                    self.writer.add_scalar("loss", loss.item(), current_iteration)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                if epoch % self.print_freq == 0:
                    self.actor.eval()
                    self.env.eval()
                    self.eval(epoch)
                    self.actor.train()
                    self.env.train()

                    torch.save(
                        self.actor.cpu().state_dict(),
                        os.path.join(self.log_dir, f"diffusion_{epoch}_epochs.pt"),
                    )
                    self.actor.to(self.device)
