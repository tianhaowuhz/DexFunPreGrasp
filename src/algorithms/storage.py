import warnings
from typing import Dict, Iterator, List, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler

__all__ = ["RolloutStorage"]


class MetaStorage(type):
    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class Storage(object, metaclass=MetaStorage):
    """Storage class for storing and managing data during training.

    Args:
        num_envs (int): Number of environments.
        buffer_size (int): Size of the buffer.
        observation_shape (Union[int, Tuple[int]]): Shape of the observation tensor.
        state_shape (Union[int, Tuple[int]]): Shape of the state tensor.
        action_shape (Union[int, Tuple[int]]): Shape of the action tensor.
        device (Union[str, torch.device], optional): Device to store the tensors on. Defaults to "cpu".

    Attributes:
        components (Tuple[str, ...]): Tuple of component names.
        observations (torch.Tensor): Tensor to store observations.
        states (torch.Tensor): Tensor to store states.
        actions (torch.Tensor): Tensor to store actions.
        rewards (torch.Tensor): Tensor to store rewards.
        dones (torch.Tensor): Tensor to store dones.

    Methods:
        __init__: Initializes the Storage object.
        __post_init__: Performs post-initialization checks and prints information about the object.
        add_transitions: Adds transitions to the storage.
        clear: Clears the storage.
        compute_statistics: Computes statistics from the stored data.
        create_dataloader: Creates a DataLoader for the stored data.
        iterator: Creates an iterator for the stored data.
        random_sample: Randomly samples a batch of data from the stored data.
    """

    components: Tuple[str, ...] = ("observations", "states", "actions", "rewards", "dones")

    observations: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        buffer_size: int,
        observation_shape: Union[int, Tuple[int]],
        state_shape: Union[int, Tuple[int]],
        action_shape: Union[int, Tuple[int]],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initializes the Storage object.

        Args:
            num_envs (int): Number of environments.
            buffer_size (int): Size of the buffer.
            observation_shape (Union[int, Tuple[int]]): Shape of the observation tensor.
            state_shape (Union[int, Tuple[int]]): Shape of the state tensor.
            action_shape (Union[int, Tuple[int]]): Shape of the action tensor.
            device (Union[str, torch.device], optional): Device to store the tensors on. Defaults to "cpu".
        """
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.device = device
        self.cursor = 0
        self.size = 0

        if self.device != "cpu":
            warnings.warn("Using GPU for RolloutStorage is not recommended.")

        observation_shape = (observation_shape,) if isinstance(observation_shape, int) else observation_shape
        state_shape = (state_shape,) if isinstance(state_shape, int) else state_shape
        action_shape = (action_shape,) if isinstance(action_shape, int) else action_shape

        self.observation_shape = observation_shape
        self.state_shape = state_shape
        self.action_shape = action_shape

        # Core Components
        self.observations = torch.zeros(buffer_size, num_envs, *observation_shape, device=self.device)
        self.states = torch.zeros(buffer_size, num_envs, *state_shape, device=self.device)
        self.actions = torch.zeros(buffer_size, num_envs, *action_shape, device=self.device)

        self.rewards = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.dones = torch.zeros(buffer_size, num_envs, 1, device=self.device).byte()

    def __post_init__(self) -> None:
        """Performs post-initialization checks and prints information about the object."""
        print(f">>> Initializing {self.__class__.__name__}")
        print(f"  - num_envs: {self.num_envs}")
        print(f"  - buffer_size: {self.buffer_size}")
        print("  - components:")
        for name in self.components:
            assert hasattr(self, name), f"Missing component {name}"
            buffer: torch.Tensor = getattr(self, name)
            assert isinstance(buffer, torch.Tensor), f"Component {name} is not a torch.Tensor"
            memory_usage = buffer.element_size() * buffer.nelement() / (1024**2)
            print(f"    - {name}: {buffer.shape}, Mem Usage: {memory_usage:.2f} MB")

    def add_transitions(self, **kwargs: torch.Tensor) -> None:
        """Adds transitions to the storage.

        Args:
            **kwargs (torch.Tensor): Transitions to be added. Each keyword argument should correspond to a component name.
        """
        for name in self.components:
            if name not in kwargs:
                continue
            tensor: torch.Tensor = kwargs[name].detach().to(self.device)
            buffer: torch.Tensor = getattr(self, name)
            buffer[self.cursor].copy_(tensor.view(-1, *buffer.shape[2:]))

        self.cursor = (self.cursor + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def clear(self) -> None:
        """Clears the storage."""
        self.cursor = 0
        self.size = 0

    def compute_statistics(self) -> Tuple[float, float]:
        """Computes statistics from the stored data.

        Returns:
            Tuple[float, float]: Mean trajectory length and mean reward.
        """
        mean_trajectory_length = (self.num_envs * self.size) / (self.dones.cpu().sum().item() + self.num_envs)
        mean_reward = self.rewards.cpu().mean().item()
        return mean_trajectory_length, mean_reward

    def create_dataloader(self, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
        """Creates a PyTorch DataLoader for the stored data.

        Args:
            batch_size (int): Batch size.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.

        Returns:
            DataLoader: DataLoader object.
        """
        assert (
            self.size * self.num_envs >= batch_size
        ), f"RolloutStorage size ({self.size * self.num_envs}) is smaller than batch_size ({batch_size})"

        tensors: List[torch.Tensor] = []
        for name in self.components:
            tensor: torch.Tensor = getattr(self, name)[: self.size]
            tensors.append(tensor.view(-1, *tensor.shape[2:]))
        dataset: Dataset = TensorDataset(*tensors)

        def _collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Dict[str, torch.Tensor]:
            return {name: torch.stack([x[i] for x in batch]) for i, name in enumerate(self.components)}

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=_collate_fn)

    def iterator(
        self, batch_size: int, shuffle: bool = False, drop_last: bool = False
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Creates a PyTorch DataLoader for the stored data.

        (directly returns an iterator instead of a DataLoader, faster but less flexible)

        Args:
            batch_size (int): Batch size.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.

        Returns:
            Iterator[Dict[str, torch.Tensor]]: Iterator object.
        """
        assert (
            self.size * self.num_envs >= batch_size
        ), f"RolloutStorage size ({self.size * self.num_envs}) is smaller than batch_size ({batch_size})"

        Sampler = SequentialSampler if not shuffle else SubsetRandomSampler
        sampler = BatchSampler(Sampler(range(self.size * self.num_envs)), batch_size, drop_last)

        for indices in sampler:
            tensors: Dict[str, torch.Tensor] = {}
            for name in self.components:
                tensor: torch.Tensor = getattr(self, name)
                tensors[name] = tensor.view(-1, *tensor.shape[2:])[indices]
            yield tensors

    def random_sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Randomly samples a batch of data from the stored data.

        Args:
            batch_size (int): Batch size.

        Returns:
            Dict[str, torch.Tensor]: A batch of data.
        """
        assert (
            self.size * self.num_envs >= batch_size
        ), f"RolloutStorage size ({self.size * self.num_envs}) is smaller than batch_size ({batch_size})"

        indices = torch.randint(0, self.size * self.num_envs, (batch_size,))
        tensors: Dict[str, torch.Tensor] = {}
        for name in self.components:
            tensor: torch.Tensor = getattr(self, name)
            tensors[name] = tensor.view(-1, *tensor.shape[2:])[indices]
        return tensors


class RolloutStorage(Storage):
    components: Tuple[str, ...] = ("observations", "states", "actions", "rewards", "dones")

    def add_transitions(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        super().add_transitions(observations=observations, states=states, actions=actions, rewards=rewards, dones=dones)


class PpoStorage(Storage):
    components: Tuple[str, ...] = (
        "observations",
        "states",
        "actions",
        "rewards",
        "dones",
        "values",
        "actions_log_prob",
        "mu",
        "sigma",
        "returns",
        "advantages",
    )

    values: torch.Tensor
    actions_log_prob: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor

    returns: torch.Tensor
    advantages: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        buffer_size: int,
        observation_shape,
        state_shape,
        action_shape,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(num_envs, buffer_size, observation_shape, state_shape, action_shape, device=device)

        self.actions_log_prob = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.values = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.mu = torch.zeros(buffer_size, num_envs, *self.action_shape, device=self.device)
        self.sigma = torch.zeros(buffer_size, num_envs, *self.action_shape, device=self.device)

        self.returns = torch.zeros(buffer_size, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(buffer_size, num_envs, 1, device=self.device)

    def add_transitions(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        actions_log_prob: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> None:
        super().add_transitions(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            values=values,
            actions_log_prob=actions_log_prob,
            mu=mu,
            sigma=sigma,
        )

    def compute_advantages(self, last_values: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        last_values = last_values.detach().to(self.device)
        advantage = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
