import copy
import os
import statistics
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate

from .algo import Algorithm
from .common.network import ActorCritic
from .storage import PpoStorage


class PPO(Algorithm):
    def __init__(
        self,
        env,
        model_cfg,
        learn_cfg,
        is_testing=False,
        device="cpu",
        log_dir="run",
        print_log=True,
    ):
        super().__init__(env, log_dir=log_dir, device=device)

        self.is_testing = is_testing
        self.model_cfg = copy.deepcopy(model_cfg)
        self.learn_cfg = copy.deepcopy(learn_cfg)

        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)

        self.num_transitions_per_env = learn_cfg["num_transitions_per_env"]
        self.learning_rate = learn_cfg["learning_rate"]
        self.step_size = learn_cfg["learning_rate"]

        observation_subspace = model_cfg.get("space", None)
        if observation_subspace is None:
            observation_metainfo = self.observation_metainfo
        else:
            observation_metainfo = []
            for name in observation_subspace:
                for info in self.observation_metainfo:
                    if info["name"] == name:
                        observation_metainfo.append(info)
                        break
                else:
                    raise ValueError(f"Observation subspace {name} not found in observation metainfo")

        action_metainfo = self.action_metainfo

        print(model_cfg)
        # PPO components
        self.actor = ActorCritic(
            observation_metainfo=observation_metainfo,
            action_metainfo=action_metainfo,
            actor_partial=instantiate(model_cfg.actor_partial),
            critic_partial=instantiate(model_cfg.critic_partial),
            initial_std=self.init_noise_std,
        )
        self.actor.to(self.device)
        self.storage = PpoStorage(
            self.num_envs,
            self.num_transitions_per_env,
            (self.num_observations,),
            (self.num_states,),
            (self.num_actions,),
            device=self.device,
        )
        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        # PPO parameters
        self.clip_param = learn_cfg["clip_range"]
        self.num_learning_epochs = learn_cfg["num_learning_epochs"]
        mini_batch_size = learn_cfg.get("mini_batch_size", None)
        num_mini_batches = learn_cfg.get("num_mini_batches", None)
        assert not (
            mini_batch_size is None and num_mini_batches is None
        ), "Either mini_batch_size or num_mini_batches must be provided"
        assert (
            mini_batch_size is None or num_mini_batches is None
        ), "Only one of mini_batch_size or num_mini_batches must be provided"
        if mini_batch_size is None:
            self.mini_batch_size = self.num_transitions_per_env * self.num_envs // num_mini_batches
        else:
            self.mini_batch_size = mini_batch_size
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg["entropy_coef"]
        self.gamma = learn_cfg["gamma"]
        self.lam = learn_cfg["lam"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False)

        # Log
        self.print_log = print_log
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def test(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def load(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor.train()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.env.reset()["obs"]
        current_states = self.env.get_state()

        if self.is_testing:
            while True:
                with torch.no_grad():
                    # Compute the action
                    actions = self.actor.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.env.step(actions)
                    current_obs.copy_(next_obs)
        else:
            self.env.train()
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                with torch.no_grad():
                    for _ in range(self.num_transitions_per_env):
                        # Compute the action
                        actions, actions_log_prob, values, mu, sigma = self.actor.act(current_obs)
                        # Step the vec_environment
                        next_obs, rews, dones, infos = self.env.step(actions)
                        next_obs = next_obs["obs"]
                        next_states = self.env.get_state()
                        # Record the transition
                        self.storage.add_transitions(
                            current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma
                        )
                        current_obs.copy_(next_obs)
                        current_states.copy_(next_states)
                        # Book keeping
                        ep_infos.append(infos)

                        if self.print_log:
                            cur_reward_sum[:] += rews
                            cur_episode_length[:] += 1

                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _ = self.actor.act(current_obs)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.compute_statistics()

                # Learning step
                start = stop
                self.storage.compute_advantages(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.eval(it)
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor.log_std.exp().mean()

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        self.writer.add_scalar("Train2/mean_reward/step", locs["mean_reward"], locs["it"])
        self.writer.add_scalar("Train2/mean_episode_length/episode", locs["mean_trajectory_length"], locs["it"])

        fps = int(self.num_transitions_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0

        for _ in range(self.num_learning_epochs):
            for batch in self.storage.iterator(self.mini_batch_size, shuffle=True):
                num_updates += 1
                obs_batch = batch["observations"].to(self.device)
                states_batch = batch["states"].to(self.device)
                actions_batch = batch["actions"].to(self.device)
                target_values_batch = batch["values"].to(self.device)
                advantages_batch = batch["advantages"].to(self.device)
                returns_batch = batch["returns"].to(self.device)
                old_actions_log_prob_batch = batch["actions_log_prob"].to(self.device)
                old_mu_batch = batch["mu"].to(self.device)
                old_sigma_batch = batch["sigma"].to(self.device)

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor.evaluate(
                    obs_batch, states_batch, actions_batch
                )

                # KL
                if self.desired_kl != None and self.schedule == "adaptive":
                    kl = torch.sum(
                        sigma_batch
                        - old_sigma_batch
                        + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        return mean_value_loss, mean_surrogate_loss
