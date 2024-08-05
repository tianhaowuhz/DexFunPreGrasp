import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MemmapTrajectoriesDataset(Dataset):
    def __init__(self, data_dir: os.PathLike, horizon: int = 4, squeeze_output: bool = False):
        data_dir: Path = Path(data_dir)
        self.horizon = horizon
        self.squeeze_output = squeeze_output and horizon == 1

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
        self.cumsum_trajectory_lengths = np.concatenate([[0], np.cumsum(self.num_datapoints)])
        self.trajectory_indices = np.concatenate(
            [np.ones(n, dtype=np.int32) * i for i, n in enumerate(self.num_datapoints)]
        )
        self.num_samples = self.num_datapoints.sum()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        trajectory_index = self.trajectory_indices[index]
        index = index - self.cumsum_trajectory_lengths[trajectory_index - 1] if trajectory_index > 0 else index

        observations = torch.from_numpy(self.observations[index : index + self.horizon].copy())
        actions = torch.from_numpy(self.actions[index : index + self.horizon].copy())

        if self.squeeze_output:
            observations, actions = observations.squeeze(0), actions.squeeze(0)

        return observations, actions


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
    observations.flush()
    actions.flush()

    metainfo = {"num_timesteps": num_timesteps, "num_observations": num_observations, "num_actions": num_actions}

    with open(target_dir / "metainfo.json", "w") as f:
        json.dump(metainfo, f)


if __name__ == "__main__":
    datadir = Path("data/example_dataset/train")

    memmap_datadir = Path("data/example_dataset/memmap")

    # create_memmap_dataset(datadir, memmap_datadir)

    dataset = MemmapTrajectoriesDataset(memmap_datadir)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    for i, (obs, act) in tqdm(enumerate(dataloader)):
        print(i, obs.shape, act.shape)

    # filepaths = sorted(list(datadir.glob("*.npy")))

    # num_timesteps = 0
    # num_observations = 0
    # num_actions = 0

    # for filepath in tqdm(filepaths):
    #     data = np.load(filepath, allow_pickle=True).tolist()

    #     observations = data['obs']
    #     actions = data['action']

    #     assert num_observations == 0 or num_observations == observations.shape[1]
    #     assert num_actions == 0 or num_actions == actions.shape[1]

    #     num_timesteps += observations.shape[0]
    #     num_observations = observations.shape[1]
    #     num_actions = actions.shape[1]

    # # observations = np.memmap("/root/projects/func-mani/data/expert_dataset_synthetic_fixed/memmap/observations.npy",
    # #                          dtype=np.float32, mode='w+', shape=(num_timesteps, num_observations))
    # # actions = np.memmap("/root/projects/func-mani/data/expert_dataset_synthetic_fixed/memmap/actions.npy",
    # #                     dtype=np.float32, mode='w+', shape=(num_timesteps, num_actions))

    # # offset = 0
    # # for filepath in tqdm(filepaths):
    # #     data = np.load(filepath, allow_pickle=True).tolist()

    # #     observations_ = data['obs']
    # #     actions_ = data['action']

    # #     observations[offset:offset+observations_.shape[0]] = observations_
    # #     actions[offset:offset+actions_.shape[0]] = actions_

    # #     offset += observations_.shape[0]

    # # observations.flush()
    # # actions.flush()

    # dataset = MemmapTrajectoriesDataset(num_timesteps,
    #     num_observations,num_actions,
    #     "/root/projects/func-mani/data/expert_dataset_synthetic_fixed/memmap/observations.npy",
    #                                 "/root/projects/func-mani/data/expert_dataset_synthetic_fixed/memmap/actions.npy")

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # print(len(dataset))

    # for i, (obs, act) in tqdm(enumerate(dataloader)):
    #     print(i, obs.shape, act.shape)
