import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MemmapTrajectoriesDataset(Dataset):
    def __init__(
        self,
        data_dir: os.PathLike,
        horizon: int = 4,
        squeeze_output: bool = False,
        postprocess_func: Optional[Callable] = None,
        max_num_trajectories_per_target: Optional[int] = None,
        num_repeats: int = 1,
    ):
        print(f"Creating MemmapTrajectoriesDataset from {data_dir}")

        data_dir: Path = Path(data_dir)
        self.horizon = horizon
        self.squeeze_output = squeeze_output and horizon == 1
        self.postprocess_func = postprocess_func
        self.max_num_trajectories_per_target = max_num_trajectories_per_target
        self.num_repeats = num_repeats

        with open(data_dir / "metainfo.json", "r") as f:
            metainfo = json.load(f)

        num_timesteps = metainfo["num_timesteps"]
        num_observations = metainfo["num_observations"]
        num_actions = metainfo["num_actions"]

        observation_memmap_filepath = data_dir / "observations.memmap"
        action_memmap_filepath = data_dir / "actions.memmap"

        self.observations = np.memmap(
            observation_memmap_filepath, dtype=np.float32, mode="r", shape=(num_timesteps, num_observations)
        )
        self.actions = np.memmap(action_memmap_filepath, dtype=np.float32, mode="r", shape=(num_timesteps, num_actions))
        self.trajectory_lengths: np.ndarray = np.load(data_dir / "trajectory_lengths.npy")
        assert self.trajectory_lengths.ndim == 1
        assert self.trajectory_lengths.sum() == num_timesteps

        self.num_datapoints = np.maximum(self.trajectory_lengths - self.horizon + 1, 0)

        if self.max_num_trajectories_per_target is not None:
            counter_per_target: np.ndarray = np.load(data_dir / "counter_per_target.npy")
            assert counter_per_target.ndim == 1
            mask = counter_per_target <= self.max_num_trajectories_per_target
            self.num_datapoints = self.num_datapoints * mask

        cumsum_datapoints = np.concatenate([[0], np.cumsum(self.num_datapoints)])
        cumsum_trajectory_lengths = np.concatenate([[0], np.cumsum(self.trajectory_lengths)])
        self.offsets = cumsum_trajectory_lengths - cumsum_datapoints
        self.trajectory_indices = np.concatenate(
            [np.ones(n, dtype=np.int32) * i for i, n in enumerate(self.num_datapoints)]
        )
        self.num_samples = self.num_datapoints.sum()

        print("self.num_samples", self.num_samples)
        print("self.num_datapoints", self.num_datapoints)
        print("self.trajectory_indices", self.trajectory_indices)
        print("self.trajectory_indices.shape", self.trajectory_indices.shape)
        print("self.offsets", self.offsets)
        print("self.offsets.shape", self.offsets.shape)
        print("self.trajectory_lengths", self.trajectory_lengths)
        print("self.trajectory_lengths.shape", self.trajectory_lengths.shape)

    def __len__(self):
        return self.num_samples * self.num_repeats

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = index % self.num_samples
        trajectory_index = self.trajectory_indices[index]
        index = index + self.offsets[trajectory_index]

        observations = torch.from_numpy(self.observations[index : index + self.horizon].copy())
        actions = torch.from_numpy(self.actions[index : index + self.horizon].copy())

        if self.squeeze_output:
            observations, actions = observations.squeeze(0), actions.squeeze(0)

        if self.postprocess_func is not None:
            observations, actions = self.postprocess_func(observations, actions)

        return observations, actions


class IladMemmapTrajectoriesDataset(Dataset):
    def __init__(
        self,
        data_dir: os.PathLike,
        horizon: int = 1,
        squeeze_output: bool = False,
        postprocess_func: Optional[Callable] = None,
        action_mode="rel",
        action_type="all",
        obs_info=None,
        pcl_number=1024,
    ):
        print(f"Creating MemmapTrajectoriesDataset from {data_dir}")
        self.action_mode = action_mode
        self.action_type = action_type
        self.obs_info = obs_info
        self.obs_tactile_dim = 14
        self.fingertip_from_all = False
        if self.fingertip_from_all:
            self.obs_tactile_dim = 5
            self.full_obs_tactile_dim = 14
        self.obs_state_dim = 208 + self.obs_tactile_dim
        self.pcl_number = pcl_number
        self.obs_pcl_dim = self.pcl_number * 3

        data_dir: Path = Path(data_dir)
        self.horizon = horizon
        self.squeeze_output = squeeze_output and horizon == 1
        self.postprocess_func = postprocess_func

        with open(data_dir / "metainfo.json", "r") as f:
            metainfo = json.load(f)

        num_timesteps = metainfo["num_timesteps"]
        num_observations = metainfo["num_observations"]
        num_actions = metainfo["num_actions"]

        observation_memmap_filepath = data_dir / "observations.memmap"
        action_memmap_filepath = data_dir / "actions.memmap"

        self.max_num_trajectories_per_target = 10

        self.observations = np.memmap(
            observation_memmap_filepath, dtype=np.float32, mode="r", shape=(num_timesteps, num_observations)
        )
        self.actions = np.memmap(action_memmap_filepath, dtype=np.float32, mode="r", shape=(num_timesteps, num_actions))
        self.trajectory_lengths: np.ndarray = np.load(data_dir / "trajectory_lengths.npy")
        assert self.trajectory_lengths.ndim == 1
        assert self.trajectory_lengths.sum() == num_timesteps

        self.num_datapoints = np.maximum(self.trajectory_lengths - self.horizon + 1, 0)

        if self.max_num_trajectories_per_target is not None:
            counter_per_target: np.ndarray = np.load(data_dir / "counter_per_target.npy")
            assert counter_per_target.ndim == 1
            mask = counter_per_target <= self.max_num_trajectories_per_target
            self.num_datapoints = self.num_datapoints * mask

        cumsum_datapoints = np.concatenate([[0], np.cumsum(self.num_datapoints)])
        cumsum_trajectory_lengths = np.concatenate([[0], np.cumsum(self.trajectory_lengths)])
        self.offsets = cumsum_trajectory_lengths - cumsum_datapoints
        self.trajectory_indices = np.concatenate(
            [np.ones(n, dtype=np.int32) * i for i, n in enumerate(self.num_datapoints)]
        )
        self.num_samples = self.num_datapoints.sum()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        trajectory_index = self.trajectory_indices[index]
        index = index + self.offsets[trajectory_index]

        observations = torch.from_numpy(self.observations[index : index + self.horizon].copy())
        actions = torch.from_numpy(self.actions[index : index + self.horizon].copy())

        if self.postprocess_func is not None:
            observations, actions = self.postprocess_func(observations, actions)

        observations = self.fetch_train_obs(observations)

        if self.squeeze_output:
            observations, actions = observations.squeeze(0), actions.squeeze(0)

        return observations, actions

    def get_random_tuple(self, indices):
        observations = self.observations[indices].copy()
        actions = self.actions[indices].copy()

        observations = self.fetch_train_obs(observations)

        observations = torch.from_numpy(observations)
        actions = torch.from_numpy(actions)

        if self.postprocess_func is not None:
            observations, actions = self.postprocess_func(observations, actions)

        # if self.squeeze_output:
        #     observations, actions = observations.squeeze(0), actions.squeeze(0)

        return observations, actions

    def fetch_train_obs(self, obs):
        if self.fingertip_from_all:
            # reorginize obs
            state_obs = self.fetch_state_from_obs_info(
                obs[:, : self.obs_state_dim - self.obs_tactile_dim]
            )  # magic number
            tactile_obs = obs[
                :,
                self.obs_state_dim
                - self.obs_tactile_dim : self.obs_state_dim
                - self.obs_tactile_dim
                + self.full_obs_tactile_dim,
            ]
            pcl_obs = obs[:, self.obs_state_dim - self.obs_tactile_dim + self.full_obs_tactile_dim :]
            assert pcl_obs.shape[1] == self.obs_pcl_dim

            # get fingertip obs from all obs
            tactile_obs = tactile_obs[:, self.ft_idx_in_all]

            obs = np.concatenate([state_obs, tactile_obs, pcl_obs], axis=1)
        else:
            state_obs = self.fetch_state_from_obs_info(
                obs[:, : self.obs_state_dim - self.obs_tactile_dim]
            )  # magic number
            other_obs = obs[:, self.obs_state_dim - self.obs_tactile_dim :]
            obs = np.concatenate([state_obs, other_obs], axis=1)
        return obs

    def fetch_state_from_obs_info(self, obs):
        self.encode_state_dim = 0
        for info in self.obs_info:
            if info["name"] == "ur_endeffector_position":
                encode_state_obs = obs[:, :7]
                self.encode_state_dim = self.encode_state_dim + 7
            if info["name"] == "shadow_hand_dof_position":
                encode_state_obs = np.concatenate([encode_state_obs, obs[:, 7:37]], axis=-1)
                self.encode_state_dim = self.encode_state_dim + 30
            if info["name"] == "object_position_wrt_palm":
                encode_state_obs = np.concatenate([encode_state_obs, obs[:, 132:139]], axis=-1)
                self.encode_state_dim = self.encode_state_dim + 7
            if info["name"] == "object_target_relposecontact":
                encode_state_obs = np.concatenate([encode_state_obs, obs[:, 152:177]], axis=-1)
                self.encode_state_dim = self.encode_state_dim + 25
        return encode_state_obs


def create_memmap_dataset(source_dir: os.PathLike, target_dir: os.PathLike):
    source_dir: Path = Path(source_dir)
    target_dir: Path = Path(target_dir)

    assert source_dir.exists() and source_dir.is_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(list(source_dir.glob("*.npy")))

    num_trajectories = len(filepaths)
    num_timesteps = 0
    num_observations = 0
    num_actions = 0

    trajectory_lengths: np.ndarray = np.zeros(num_trajectories, dtype=np.int32)
    counter_per_target: np.ndarray = np.zeros(num_trajectories, dtype=np.int32)

    counter = {}
    for i, filepath in enumerate(tqdm(filepaths)):
        data = np.load(filepath, allow_pickle=True).tolist()

        observations = data["obs"]
        actions = data["action"]

        assert num_observations == 0 or num_observations == observations.shape[1]
        assert num_actions == 0 or num_actions == actions.shape[1]

        trajectory_lengths[i] = observations.shape[0]
        num_timesteps += observations.shape[0]
        num_observations = observations.shape[1]
        num_actions = actions.shape[1]

        counter[data["object_grasp"]] = counter.get(data["object_grasp"], 0) + 1
        counter_per_target[i] = counter[data["object_grasp"]]

    observations = np.memmap(
        target_dir / "observations.memmap", dtype=np.float32, mode="w+", shape=(num_timesteps, num_observations)
    )
    actions = np.memmap(target_dir / "actions.memmap", dtype=np.float32, mode="w+", shape=(num_timesteps, num_actions))

    offset = 0
    for filepath in tqdm(filepaths):
        data = np.load(filepath, allow_pickle=True).tolist()

        observations_ = data["obs"]
        actions_ = data["action"]

        observations[offset : offset + observations_.shape[0]] = observations_
        actions[offset : offset + actions_.shape[0]] = actions_

        offset += observations_.shape[0]

    np.save(target_dir / "trajectory_lengths.npy", trajectory_lengths)
    np.save(target_dir / "counter_per_target.npy", counter_per_target)

    observations.flush()
    actions.flush()

    metainfo = {"num_timesteps": num_timesteps, "num_observations": num_observations, "num_actions": num_actions}

    with open(target_dir / "metainfo.json", "w") as f:
        json.dump(metainfo, f)
