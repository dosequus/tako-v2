"""Replay Buffer for storing and sampling training data."""

import numpy as np
import torch
from typing import List, Dict, Tuple
import random


class ReplayBuffer:
    """Circular buffer for storing self-play game samples.

    Stores (state, policy, value) tuples from self-play games.
    Implements uniform random sampling for training batches.

    Attributes:
        capacity: Maximum number of samples to store
        min_size: Minimum samples before training can start
        states: List of tokenized game states
        policies: List of MCTS policy targets
        values: List of game outcomes
    """

    def __init__(self, capacity: int, min_size: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum buffer size (e.g., 1M positions)
            min_size: Minimum size before buffer is ready for training
        """
        self.capacity = capacity
        self.min_size = min_size

        # Storage
        self.states: List[torch.Tensor] = []
        self.policies: List[np.ndarray] = []
        self.values: List[float] = []

        self._write_pos = 0  # Position for circular buffer writes

    def add_samples(self, samples: List[Dict]):
        """Add samples to the buffer.

        If buffer is at capacity, oldest samples are overwritten (circular).

        Args:
            samples: List of dicts, each with keys:
                - 'state': torch.Tensor of tokens [seq_len]
                - 'policy': np.ndarray of visit counts [action_size]
                - 'value': float in {-1.0, 0.0, 1.0}
        """
        for sample in samples:
            state = sample['state']
            policy = sample['policy']
            value = sample['value']

            # Validate
            assert isinstance(state, torch.Tensor), "State must be a torch.Tensor"
            assert isinstance(policy, np.ndarray), "Policy must be a numpy array"
            assert isinstance(value, (int, float)), "Value must be a number"
            assert value in [-1.0, 0.0, 1.0], f"Value must be in {{-1, 0, 1}}, got {value}"

            if len(self.states) < self.capacity:
                # Buffer not full yet: append
                self.states.append(state)
                self.policies.append(policy)
                self.values.append(value)
            else:
                # Buffer full: overwrite oldest (circular)
                self.states[self._write_pos] = state
                self.policies[self._write_pos] = policy
                self.values[self._write_pos] = value

                # Advance write position (circular)
                self._write_pos = (self._write_pos + 1) % self.capacity

    def sample_batch(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch for training.

        Args:
            batch_size: Number of samples to draw

        Returns:
            Tuple of (states, policies, values):
                - states: [batch_size, seq_len] tensor
                - policies: [batch_size, action_size] tensor
                - values: [batch_size] tensor
        """
        if len(self.states) < batch_size:
            raise ValueError(
                f"Buffer has {len(self.states)} samples, cannot sample batch of {batch_size}"
            )

        # Uniform random sampling
        indices = random.sample(range(len(self.states)), batch_size)

        # Gather samples
        batch_states = [self.states[i] for i in indices]
        batch_policies = [self.policies[i] for i in indices]
        batch_values = [self.values[i] for i in indices]

        # Stack into tensors
        # States: pad to max length in batch
        max_len = max(s.shape[0] for s in batch_states)
        states_padded = []
        for state in batch_states:
            if state.shape[0] < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - state.shape[0], dtype=state.dtype)
                state = torch.cat([state, padding])
            states_padded.append(state)

        states_tensor = torch.stack(states_padded)  # [batch_size, seq_len]
        policies_tensor = torch.from_numpy(np.stack(batch_policies)).float()  # [batch_size, action_size]
        values_tensor = torch.tensor(batch_values, dtype=torch.float32)  # [batch_size]

        return states_tensor, policies_tensor, values_tensor

    def ready(self) -> bool:
        """Check if buffer has enough samples to start training.

        Returns:
            True if buffer size >= min_size
        """
        return len(self.states) >= self.min_size

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.states)

    def stats(self) -> Dict:
        """Return buffer statistics.

        Returns:
            Dict with keys:
                - size: Current number of samples
                - capacity: Maximum capacity
                - utilization: Fraction full (0.0 to 1.0)
                - ready: Whether buffer is ready for training
        """
        size = len(self.states)
        return {
            'size': size,
            'capacity': self.capacity,
            'utilization': size / self.capacity if self.capacity > 0 else 0.0,
            'ready': self.ready(),
            'min_size': self.min_size
        }
