import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gym.spaces import Space
from isaacgymenvs.tasks.base.vec_task import VecTask
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tasks.isaacgym_utils import images_to_video, pack_pointcloud_observations


class CircularBuffer:
    """Circular buffer for storing and updating data in a rolling window fashion.

    Args:
        num_envs (int): Number of environments.
        num_channels (int): Number of channels in the data.
        num_steps (int): Number of steps to store in the buffer.
        stride (int, optional): Stride for downsampling the buffer. Defaults to 1.
        dtype (torch.dtype, optional): Data type of the buffer. Defaults to torch.float32.
        device (torch.device, optional): Device to store the buffer on. Defaults to "cpu".

    Attributes:
        num_envs (int): Number of environments.
        num_channels (int): Number of channels in the data.
        num_steps (int): Number of steps to store in the buffer.
        stride (int): Stride for downsampling the buffer.
        dtype (torch.dtype): Data type of the buffer.
        device (torch.device): Device to store the buffer on.
        buffer (torch.Tensor): Buffer to store the data.
        reset_buf (torch.Tensor): Buffer to track reset flags.

    Methods:
        update(data: torch.Tensor, dones: Optional[torch.Tensor] = None) -> None:
            Update the buffer with new data and reset flags.

        get() -> torch.Tensor:
            Get the current data from the buffer.

        reset() -> None:
            Reset the buffer and reset flags.
    """

    def __init__(
        self,
        num_envs: int,
        num_channels: int,
        num_steps: int = 1,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
        squeeze_output: bool = True,
    ):
        self.num_envs = num_envs
        self.num_channels = num_channels
        self.num_steps = num_steps
        self.stride = stride
        self.dtype = dtype
        self.device = device
        self.squeeze_output = squeeze_output

        self.buffer = torch.zeros(num_envs, num_steps * stride, num_channels, dtype=dtype, device=device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self.reset()

    def update(self, data: torch.Tensor, dones: Optional[torch.Tensor] = None) -> None:
        """Update the buffer with new data and reset flags.

        Args:
            data (torch.Tensor): New data to be added to the buffer.
            dones (torch.Tensor, optional): Reset flags for the environments. Defaults to None.
        """
        data = data.to(self.device)
        if dones is not None:
            dones = dones.to(self.device)

        if self.reset_buf.any():
            self.buffer[self.reset_buf, :, :] = data[self.reset_buf, None, :]
            self.reset_buf[:] = False

        self.buffer[:, :-1, :] = self.buffer[:, 1:, :]
        self.buffer[:, -1, :] = data

        if dones is not None:
            self.reset_buf[dones > 0] = True

    def get(self) -> torch.Tensor:
        """Get the current data from the buffer.

        Returns:
            torch.Tensor: Current data stored in the buffer.
        """
        observation = self.buffer[:, self.stride - 1 :: self.stride, :]

        if self.squeeze_output and observation.size(1) == 1:
            observation = observation.squeeze(1)

        return observation

    def reset(self, data: Optional[torch.Tensor] = None) -> None:
        """Reset the buffer and reset flags."""
        self.buffer.zero_()
        self.reset_buf[:] = True

        if data is not None:
            self.update(data)


class Algorithm:
    env: VecTask
    actor: nn.Module
    device: torch.device
    writer: SummaryWriter

    observation_space: Space
    state_space: Space
    action_space: Space

    observation_shape: Tuple[int, ...]
    state_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]

    num_envs: int
    num_observations: int
    num_states: int
    num_actions: int
    num_observation_steps: int

    observation_metainfo: List[Dict[str, Any]]
    action_metainfo: List[Dict[str, Any]]

    def __init__(
        self,
        env: VecTask,
        experiment_name: str = "debug",
        device: torch.device = "cpu",
        print_freq: int = 1,
        num_observation_steps: int = 1,
        num_evaluation_rounds: int = 10,
        save_trajectory: bool = False,
    ):
        if not isinstance(env.observation_space, Space):
            raise TypeError("env.observation_space must be a gym Space")
        if not isinstance(env.state_space, Space):
            raise TypeError("env.state_space must be a gym Space")
        if not isinstance(env.action_space, Space):
            raise TypeError("env.action_space must be a gym Space")
        self.env = env
        self.observation_space = env.observation_space
        self.state_space = env.state_space
        self.action_space = env.action_space

        self.observation_shape = env.observation_space.shape
        self.state_shape = env.state_space.shape
        self.action_shape = env.action_space.shape

        self.num_envs = env.num_envs
        self.num_observations = env.num_observations
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length

        self.circular_buffer = CircularBuffer(
            num_envs=self.num_envs,
            num_channels=self.num_observations,
            num_steps=num_observation_steps,
            device=device,
        )
        self.num_observation_steps = num_observation_steps

        self.observation_metainfo = self.env.export_observation_metainfo()
        self.action_metainfo = self.env.export_action_metainfo()

        self.env = env
        self.experiment_name = experiment_name
        self.log_dir = self.create_logging_directory("logs", experiment_name)
        self.device = device
        self.print_freq = print_freq
        self.num_evaluation_rounds = num_evaluation_rounds
        self.save_trajectory = save_trajectory
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.vis_env_num = self.env.vis_env_num
        self.img_width = int(1024 / 2)
        self.img_height = int(768 / 2)
        if self.vis_env_num > 0:
            self.video_log_dir = os.path.join(self.log_dir, "video")
            os.makedirs(self.video_log_dir, exist_ok=True)
        if self.save_trajectory:
            self.demo_dir = os.path.join(self.log_dir, "demo")
            os.makedirs(self.demo_dir, exist_ok=True)

    @classmethod
    def create_logging_directory(cls, logging_directory: os.PathLike, experiment_name: str) -> Path:
        algorithm_name = cls.__name__
        current_timestamp = time.strftime("%m-%d-%H-%M", time.localtime())
        return Path(logging_directory) / algorithm_name / f"{current_timestamp}_{experiment_name}"

    def get_observation_subset(self, observation_space: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if observation_space is None:
            return self.observation_metainfo

        observation_info = []
        for name in observation_space:
            for info in self.observation_metainfo:
                if info["name"] == name:
                    observation_info.append(info)
                    break
            else:
                raise ValueError(f"observation {name} not found")
        return observation_info

    @staticmethod
    def parse_observation(
        observations: torch.Tensor, observation_info: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Parse observations into states and pointclouds.

        Args:
            observations (torch.Tensor): Batch of observations.
            observation_info (List[Dict[str, Any]]): Observation metainfo.

        Returns:
            Dict[str, torch.Tensor]: Parsed observations.
        """
        batch_size: int = observations.size(0)
        states = []
        pointclouds = {}

        for info in observation_info:
            if "pointcloud" in info["tags"]:
                pointclouds[info["name"]] = {}
                pointclouds[info["name"]]["points"] = observations[:, info["start"] : info["end"]].reshape(
                    batch_size, -1, 3
                )
            else:
                states.append(observations[:, info["start"] : info["end"]])

        if len(pointclouds) > 0:
            states = torch.cat(states, dim=-1)
            pointclouds = pack_pointcloud_observations(pointclouds)
            return {"states": states, "pointclouds": pointclouds}

        return {"states": torch.cat(states, dim=-1)}

    def run(self, num_learning_iterations: int, log_interval: int = 1):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Generate actions from observations.

        Args:
            observations (torch.Tensor): Batch of observations.

        Returns:
            torch.Tensor: Actions.
        """
        raise NotImplementedError

    def record(
        self,
        num_demonstrations_per_env: int = 20,
        max_num_rounds: int = 10,
        demonstration_dir: os.PathLike = "demonstrations",
    ):
        """TODO: Not tested yet."""
        # switch to evaluation mode
        self.env.eval()
        self.actor.eval()

        # create demonstration directory
        demonstration_dir: Path = Path(demonstration_dir)
        if not demonstration_dir.exists():
            demonstration_dir.mkdir(parents=True)

        num_envs: int = self.env.num_envs
        max_episode_length: int = self.env.max_episode_length
        num_observations: int = self.env.num_observations
        num_actions: int = self.env.num_actions
        counter: Dict[str, int] = {}

        # dump metainfo
        metainfo = {}
        metainfo["observation_space"] = self.env.export_observation_metainfo()
        metainfo["action_space"] = self.env.export_action_metainfo()

        with open(demonstration_dir / "metainfo.pkl", "wb") as f:
            pickle.dump(metainfo, f)

        observation_buffer = torch.zeros((max_episode_length, num_envs, num_observations), device="cpu")
        action_buffer = torch.zeros((max_episode_length, num_envs, num_actions), device="cpu")

        round = 0
        while round < max_num_rounds:
            print(f"round: {round}")
            print(f"counter: {counter}")
            # reset the environment
            curr_observation = self.env.reset()["obs"]

            done_status = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            success_status = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

            init_states = {}
            init_states["object_positions"] = self.env.occupied_object_init_root_positions.cpu()
            init_states["object_orientations"] = self.env.occupied_object_init_root_orientations.cpu()

            observation_buffer.zero_()
            action_buffer.zero_()

            for step in range(max_episode_length):
                actions = self.actor.act_inference(curr_observation)
                actions[done_status, :] = 0.0

                observation_buffer[step].copy_(curr_observation.detach().cpu())
                action_buffer[step].copy_(actions.detach().cpu())

                next_observation, rewards, dones, _ = self.env.step(actions)
                curr_observation.copy_(next_observation["obs"])
                step += 1

                done_current = (dones > 0) & (~done_status)
                if done_current.any():
                    done_status[done_current] = True
                    success_status[rewards >= self.env.reach_goal_bonus] = True

                    if not (done_current & success_status).any():
                        continue

                    for i in range(num_envs):
                        if not (done_current[i] and success_status[i]):
                            continue

                        code = self.env.occupied_object_codes[i]
                        grasp = self.env.occupied_object_grasps[i]

                        if counter.get(code, 0) >= num_demonstrations_per_env:
                            continue
                        counter[code] = counter.get(code, 0) + 1

                        demonstration = {"init": {}, "trajectory": {}}

                        demonstration["init"]["code"] = code
                        demonstration["init"]["grasp"] = grasp
                        demonstration["init"]["object_position"] = (
                            init_states["object_positions"][i, :].detach().cpu().numpy()
                        )
                        demonstration["init"]["object_orientation"] = (
                            init_states["object_orientations"][i, :].detach().cpu().numpy()
                        )

                        demonstration["trajectory"]["observation"] = (
                            observation_buffer[:step, i, :].detach().cpu().numpy()
                        )
                        demonstration["trajectory"]["action"] = action_buffer[:step, i, :].detach().cpu().numpy()

                        with open(demonstration_dir / f"{code}_{counter[code]}.pkl", "wb") as f:
                            pickle.dump(demonstration, f)

                if done_status.all():
                    break

            print(f"round: {round}")
            round += 1

    def on_validation_epoch_start(self, iteration: int) -> None:
        pass

    def eval(self, iteration: int, num_rounds: Optional[int] = None) -> None:
        num_envs: int = self.env.num_envs
        num_rounds = num_rounds if num_rounds is not None else self.num_evaluation_rounds

        # switch to evaluation mode
        training = self.env.training
        self.env.eval()

        # aggregated metrics for each round
        r_success_rates = []
        r_rewards = []
        r_episode_lengths = []
        r_success_episode_lengths = []
        r_min_pos_dist = []
        r_min_rot_dist = []
        r_min_contact_dist = []
        r_done_pos_dist = []
        r_done_rot_dist = []
        r_done_contact_dist = []

        metainfo = pd.DataFrame()

        with torch.no_grad():
            for round in tqdm(range(num_rounds), desc="Evaluating"):
                # TODO: track more trajectory-level metainfo
                self.on_validation_epoch_start(iteration)

                counter = 0

                done_status = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                success_status = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
                done_rewards = torch.zeros(num_envs, dtype=torch.float, device=self.device)
                episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=self.device)

                min_pos_dist = torch.ones(num_envs, device=self.device)
                min_rot_dist = torch.ones(num_envs, device=self.device)
                min_contact_dist = torch.ones(num_envs, device=self.device)
                done_pos_dist = torch.zeros(num_envs, device=self.device)
                done_rot_dist = torch.zeros(num_envs, device=self.device)
                done_contact_dist = torch.zeros(num_envs, device=self.device)
                high_similarities = torch.zeros(num_envs, device=self.device)

                current_obs = self.env.reset()["obs"]
                self.circular_buffer.reset(current_obs)

                if self.save_trajectory:
                    self.demo_init_state = {}
                    self.demo_init_state["obj_pos"] = self.env.occupied_object_init_root_positions.cpu().numpy()
                    self.demo_init_state["obj_orn"] = self.env.occupied_object_init_root_orientations.cpu().numpy()
                    self.demo_init_state["robot_dof"] = self.env.robot_init_dof.cpu().numpy()
                    self.demo_init_state["target_dof"] = self.env.object_targets.cpu().numpy()
                    self.demo_obs = torch.tensor([], device="cpu")
                    self.demo_action = torch.tensor([], device="cpu")
                    self.demo_pos_dist = torch.tensor([], device="cpu")
                    self.demo_rot_dist = torch.tensor([], device="cpu")
                    self.demo_contact_dist = torch.tensor([], device="cpu")

                if self.vis_env_num > 0:
                    all_images = np.zeros(
                        (self.env.num_envs, self.env.max_episode_length, self.img_height, self.img_width, 3)
                    )
                    image = self.env.get_images(img_width=self.img_width, img_height=self.img_height).reshape(
                        self.vis_env_num, self.img_height, self.img_width, 3
                    )
                    all_images[:, self.env.progress_buf[0], :, :, :] = image.cpu().numpy()

                curr_metainfo: pd.DataFrame = self.env.get_env_metainfo()

                while counter < num_envs:
                    # Compute the action
                    observations = self.circular_buffer.get().to(self.device)
                    actions = self.forward(observations)
                    actions[done_status, :] = 0

                    if self.save_trajectory:
                        full_obs = current_obs.reshape(1, self.env.num_envs, -1).clone()
                        self.demo_obs = torch.cat([self.demo_obs, full_obs.cpu()])
                        # self.demo_action = torch.cat(
                        #     [
                        #         self.demo_action,
                        #         self.env.clamped_actions.clone().reshape(1, self.env.num_envs, -1).cpu(),
                        #     ]
                        # )

                    next_obs, rewards, dones, infos = self.env.step(actions)

                    if self.save_trajectory:
                        self.demo_pos_dist = torch.cat(
                            [self.demo_pos_dist, infos["pos_dist"].reshape(1, self.env.num_envs, -1).clone().cpu()]
                        )
                        self.demo_rot_dist = torch.cat(
                            [self.demo_rot_dist, infos["rot_dist"].reshape(1, self.env.num_envs, -1).clone().cpu()]
                        )
                        self.demo_contact_dist = torch.cat(
                            [
                                self.demo_contact_dist,
                                infos["contact_dist"].reshape(1, self.env.num_envs, -1).clone().cpu(),
                            ]
                        )

                    if self.vis_env_num > 0:
                        image = self.env.get_images(img_width=self.img_width, img_height=self.img_height).reshape(
                            self.vis_env_num, self.img_height, self.img_width, 3
                        )
                        all_images[:, self.env.progress_buf[0], :, :, :] = image.cpu().numpy()

                    # target pos dist is 0.01, rot dist is 0.1, contact dist is 0.15
                    # compute current pos dist, rot dist, contact dist how close to target
                    pos_similarity = 0.01 / (infos["pos_dist"] + 0.01)
                    rot_similarity = 0.1 / (infos["rot_dist"] + 0.1)
                    contact_similarity = 0.15 / (infos["contact_dist"] + 0.15)
                    all_similarity = pos_similarity * rot_similarity * contact_similarity
                    # get env id which is higher than current high_similarities
                    higher_env_ids = (all_similarity > high_similarities).nonzero(as_tuple=False).squeeze(-1)
                    min_pos_dist[higher_env_ids] = infos["pos_dist"][higher_env_ids]
                    min_rot_dist[higher_env_ids] = infos["rot_dist"][higher_env_ids]
                    min_contact_dist[higher_env_ids] = infos["contact_dist"][higher_env_ids]
                    high_similarities[higher_env_ids] = all_similarity[higher_env_ids]

                    current_obs.copy_(next_obs["obs"])
                    done_current = (dones > 0) & (~done_status)

                    self.circular_buffer.update(current_obs, dones)

                    if done_current.any():
                        counter += done_current.sum().item()
                        done_rewards[done_current] = rewards[done_current]
                        done_status[done_current] = True
                        success_status[rewards >= self.env.reach_goal_bonus] = True
                        episode_lengths[done_current] = self.env.progress_buf[done_current]
                        done_pos_dist[done_current] = infos["pos_dist"][done_current]
                        done_rot_dist[done_current] = infos["rot_dist"][done_current]
                        done_contact_dist[done_current] = infos["contact_dist"][done_current]

                        if self.save_trajectory:
                            success_env_ids = (rewards >= self.env.reach_goal_bonus).nonzero(as_tuple=False).squeeze(-1)
                            new_done_env_ids = done_current.nonzero(as_tuple=False).squeeze(-1)
                            for success_env_id in success_env_ids:
                                if success_env_id in new_done_env_ids:
                                    cur_demo = {}
                                    cur_demo["env_mode"] = self.env.env_mode
                                    cur_demo["obs_space"] = self.env.cfg["env"]["observationSpace"]
                                    cur_demo["action_space"] = self.env.cfg["env"]["actionSpace"]
                                    cur_demo["ft_idx_in_all"] = self.env.ft_idx_in_all

                                    cur_demo["init_obj_pos"] = self.demo_init_state["obj_pos"][success_env_id]
                                    cur_demo["init_obj_orn"] = self.demo_init_state["obj_orn"][success_env_id]
                                    cur_demo["init_robot_dof"] = self.demo_init_state["robot_dof"][success_env_id]
                                    cur_demo["target_dof"] = self.demo_init_state["target_dof"][success_env_id]
                                    cur_demo["obs"] = self.demo_obs[:, success_env_id, :].cpu().numpy()
                                    cur_demo["pos_dist"] = self.demo_pos_dist[:, success_env_id].cpu().numpy()
                                    cur_demo["rot_dist"] = self.demo_rot_dist[:, success_env_id].cpu().numpy()
                                    cur_demo["contact_dist"] = self.demo_contact_dist[:, success_env_id].cpu().numpy()
                                    # cur_demo["action"] = self.demo_action[:, success_env_id, :].detach().cpu().numpy()

                                    # get cur object code
                                    cur_demo_object_code = self.env.occupied_object_codes[success_env_id]
                                    cur_demo["object_code"] = cur_demo_object_code
                                    cur_demo_object_grasp = self.env.occupied_object_grasps[success_env_id]
                                    cur_demo["object_grasp"] = cur_demo_object_grasp

                                    # save demo using npy
                                    np.save(
                                        os.path.join(
                                            self.demo_dir,
                                            f"traj_{cur_demo_object_code}_{cur_demo_object_grasp}_round:{round}_epl:{self.env.progress_buf[success_env_id]}.npy",
                                        ),
                                        cur_demo,
                                    )

                if self.vis_env_num > 0:
                    for vis_id in done_status.nonzero(as_tuple=False).squeeze(-1):
                        image = all_images[vis_id, : episode_lengths[vis_id] + 5, :]
                        cur_code = curr_metainfo["code"][vis_id.item()]
                        cur_grasp = curr_metainfo["grasp"][vis_id.item()]
                        save_path = os.path.join(
                            self.video_log_dir,
                            f"{cur_code}+{cur_grasp}+succ:{success_status[vis_id]}_eps:{episode_lengths[vis_id]}_envid:{vis_id}_round:{round}",
                        )
                        images_to_video(
                            path=save_path, images=image, size=(self.img_width, self.img_height), suffix="mp4"
                        )
                assert done_status.all()

                success_rate = success_status.float().mean().item()
                print(
                    f"round {round}: success rate: {success_rate * 100:.2f}%"
                    f", min_pos_dist: {min_pos_dist.float().mean().item() * 100:.2f}cm"
                    f", min_rot_dist: {min_rot_dist.float().mean().item():.2f}rad"
                    f", min_contact_dist: {min_contact_dist.float().mean().item():.2f}rad"
                    f", length: {episode_lengths.float().mean().item()}"
                )

                r_success_rates.append(success_status.float().mean().item())
                r_rewards.append(done_rewards.float().mean().item())
                r_episode_lengths.append(episode_lengths.float().mean().item())
                r_success_episode_lengths.append(episode_lengths[success_status].float().mean().item())
                r_min_pos_dist.append(min_pos_dist.float().mean().item())
                r_min_rot_dist.append(min_rot_dist.float().mean().item())
                r_min_contact_dist.append(min_contact_dist.float().mean().item())

                curr_metainfo["success"] = success_status.cpu().numpy()
                curr_metainfo["episode_length"] = episode_lengths.cpu().numpy()
                curr_metainfo["min_pos_dist"] = min_pos_dist.cpu().numpy()
                curr_metainfo["min_rot_dist"] = min_rot_dist.cpu().numpy()
                curr_metainfo["min_contact_dist"] = min_contact_dist.cpu().numpy()
                curr_metainfo["done_pos_dist"] = done_pos_dist.cpu().numpy()
                curr_metainfo["done_rot_dist"] = done_rot_dist.cpu().numpy()
                curr_metainfo["done_contact_dist"] = done_contact_dist.cpu().numpy()

                metainfo = pd.concat([metainfo, curr_metainfo], ignore_index=True, axis=0)

        metainfo.to_csv(os.path.join(self.log_dir, f"evaluation_details_{iteration}.csv"), index=False)

        # switch back to training mode
        if training:
            self.env.train()

        sr_mean, sr_std = np.mean(r_success_rates), np.std(r_success_rates)
        print(f"|| num_envs: {num_envs} || eval_times: {num_rounds}")
        print(f"eval_success_rate: {sr_mean * 100:.2f}% ± {sr_std * 100:.2f}%")
        print(f"eval_rewards: {np.mean(r_rewards)}")
        print(f"eval_eps_len: {np.mean(r_episode_lengths)}")
        print(f"eval_succ_eps_len: {np.mean(r_success_episode_lengths)}")
        print(f"eval_min_pos_dist: {np.mean(r_min_pos_dist) * 100} cm")
        print(f"eval_min_rot_dist: {np.mean(r_min_rot_dist)} rad")
        print(f"eval_min_contact_dist: {np.mean(r_min_contact_dist)} rad")
        self.writer.add_scalar("Eval/success_rate", sr_mean, iteration)
        self.writer.add_scalar("Eval/eval_rews", np.mean(r_rewards), iteration)

    def save_robot_trajectories(self, success_only: bool = True) -> None:
        num_envs: int = self.env.num_envs

        # buffer for robot states
        goal_object_positions_wrt_palm = torch.zeros((num_envs, 3), device="cpu")
        goal_object_orientations_wrt_palm = torch.zeros((num_envs, 4), device="cpu")
        goal_shadow_dof_positions = torch.zeros((num_envs, 24), device="cpu")
        endeffector_positions = torch.zeros((num_envs, self.max_episode_length, 3), device="cpu")
        endeffector_orientations = torch.zeros((num_envs, self.max_episode_length, 4), device="cpu")
        palm_positions = torch.zeros((num_envs, self.max_episode_length, 3), device="cpu")
        palm_orientations = torch.zeros((num_envs, self.max_episode_length, 4), device="cpu")
        object_positions = torch.zeros((num_envs, self.max_episode_length, 3), device="cpu")
        object_orientations = torch.zeros((num_envs, self.max_episode_length, 4), device="cpu")
        ur10e_dof_positions = torch.zeros((num_envs, self.max_episode_length, 6), device="cpu")
        shadow_dof_positions = torch.zeros((num_envs, self.max_episode_length, 24), device="cpu")

        # buffer for actions
        network_actions = torch.zeros((num_envs, self.max_episode_length, 26), device="cpu")
        clamped_actions = torch.zeros((num_envs, self.max_episode_length, 26), device="cpu")
        absolute_ur10e_targets = torch.zeros((num_envs, self.max_episode_length, 6), device="cpu")
        absolute_shadow_targets = torch.zeros((num_envs, self.max_episode_length, 24), device="cpu")

        done_status = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        success_status = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        def _save_current_frame(step: int) -> None:
            endeffector_positions[:, step, :] = self.env.endeffector_positions.clone().cpu()
            endeffector_orientations[:, step, :] = self.env.endeffector_orientations.clone().cpu()
            palm_positions[:, step, :] = self.env.shadow_hand_center_positions.clone().cpu()
            palm_orientations[:, step, :] = self.env.shadow_hand_center_orientations.clone().cpu()
            object_positions[:, step, :] = self.env.object_root_positions.clone().cpu()
            object_orientations[:, step, :] = self.env.object_root_orientations.clone().cpu()
            ur10e_dof_positions[:, step, :] = self.env.shadow_hand_dof_positions[:, :6].clone().cpu()
            shadow_dof_positions[:, step, :] = self.env.shadow_hand_dof_positions[:, 6:].clone().cpu()

            if step > 0:
                network_actions[:, step - 1, :] = self.env.actions.clone().cpu()
                clamped_actions[:, step - 1, :] = self.env.clamped_actions.clone().cpu()
                absolute_ur10e_targets[:, step - 1, :] = self.env.curr_targets[:, :6].clone().cpu()
                absolute_shadow_targets[:, step - 1, :] = self.env.curr_targets[:, 6:].clone().cpu()

        self.on_validation_epoch_start(0)
        current_obs = self.env.reset()["obs"]
        self.circular_buffer.reset(current_obs)
        goal_object_positions_wrt_palm[:] = self.env._r_target_object_positions_wrt_palm.clone().cpu()
        goal_object_orientations_wrt_palm[:] = self.env._r_target_object_orientations_wrt_palm.clone().cpu()
        goal_shadow_dof_positions[:] = self.env._r_target_shadow_dof_positions.clone().cpu()
        _save_current_frame(0)

        with torch.no_grad():
            for step in tqdm(range(self.max_episode_length)):
                observations = self.circular_buffer.get().to(self.device)
                actions = self.forward(observations)

                next_obs, rewards, dones, infos = self.env.step(actions)
                done_current = (dones > 0) & (~done_status)
                _save_current_frame(step + 1)

                current_obs.copy_(next_obs["obs"])
                self.circular_buffer.update(current_obs, dones)

                if done_current.any():
                    done_status[done_current] = True
                    success_status[rewards >= self.env.reach_goal_bonus] = True
                    episode_lengths[done_current] = self.env.progress_buf[done_current]

                if done_status.all():
                    break

        print(f"Saving robot trajectories... {success_status.sum().item()}/{num_envs} success")
        for i in range(num_envs):
            if success_only and not success_status[i]:
                continue

            trajectory = {
                "goal_object_positions_wrt_palm": goal_object_positions_wrt_palm[i, :].numpy(),
                "goal_object_orientations_wrt_palm": goal_object_orientations_wrt_palm[i, :].numpy(),
                "goal_shadow_dof_positions": goal_shadow_dof_positions[i, :].numpy(),
                "endeffector_positions": endeffector_positions[i, : episode_lengths[i]].numpy(),
                "endeffector_orientations": endeffector_orientations[i, : episode_lengths[i]].numpy(),
                "object_positions": object_positions[i, : episode_lengths[i]].numpy(),
                "object_orientations": object_orientations[i, : episode_lengths[i]].numpy(),
                "ur10e_dof_positions": ur10e_dof_positions[i, : episode_lengths[i]].numpy(),
                "shadow_dof_positions": shadow_dof_positions[i, : episode_lengths[i]].numpy(),
                "network_actions": network_actions[i, : episode_lengths[i] - 1].numpy(),
                "clamped_actions": clamped_actions[i, : episode_lengths[i] - 1].numpy(),
                "absolute_ur10e_targets": absolute_ur10e_targets[i, : episode_lengths[i] - 1].numpy(),
                "absolute_shadow_targets": absolute_shadow_targets[i, : episode_lengths[i] - 1].numpy(),
            }
            np.save(os.path.join(self.log_dir, f"robot_trajectory_{i}.npy"), trajectory)
