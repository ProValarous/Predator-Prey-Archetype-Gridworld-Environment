# src/baselines/DQN/replay_buffer.py
"""
Fixed-size circular buffer of past transitions ("ring buffer").

Preallocated numpy arrays, not a deque: numpy fancy-indexing lets
sample() pull scattered random indices in one vectorized operation
regardless of how full the buffer is. A deque gives O(1) append but
O(n) random-index access, which gets progressively slower as the
buffer fills toward capacity.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = None):
        if capacity <= 0:
            raise ValueError(f"buffer capacity must be positive, got {capacity}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.rng = np.random.default_rng(seed)

        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=bool)

        self._write_ptr = 0  # next write slot, wraps at capacity
        self._size = 0  # valid entries currently stored

    def push(self, state, action, reward, next_state, done) -> None:
        idx = self._write_ptr
        self._states[idx] = state
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._next_states[idx] = next_state
        self._dones[idx] = bool(done)

        self._write_ptr = (self._write_ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions; buffer only "
                f"has {self._size} stored so far."
            )
        idx = self.rng.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[idx].copy(),
            self._actions[idx].copy(),
            self._rewards[idx].copy(),
            self._next_states[idx].copy(),
            self._dones[idx].copy(),
        )

    def __len__(self):
        return self._size
