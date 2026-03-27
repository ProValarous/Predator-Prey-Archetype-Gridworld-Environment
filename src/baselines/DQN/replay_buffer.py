"""
Experience Replay Buffer for DQN.

This module is a pure data structure. It stores transitions
and returns random mini-batches as PyTorch tensors.

It contains:
    - NO environment logic
    - NO reward computation
    - NO state encoding
    - NO agent decision logic

Transitions are stored as raw numpy arrays. Conversion to
PyTorch tensors happens only at sample time.

Tensor types returned by sample():
    states      : torch.FloatTensor  (batch_size, state_dim)
    actions     : torch.LongTensor   (batch_size,)
    rewards     : torch.FloatTensor  (batch_size,)
    next_states : torch.FloatTensor  (batch_size, state_dim)
    dones       : torch.FloatTensor  (batch_size,)   values: 0.0 or 1.0
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity FIFO replay buffer for experience replay.

    Stores transitions as ``(state, action, reward, next_state, done)``
    tuples. When capacity is reached, the oldest transition is
    automatically discarded.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    seed : int, optional
        Seed for the internal sampling RNG. Does not affect
        any external RNG or environment state.

    Examples
    --------
    >>> buf = ReplayBuffer(capacity=10000, seed=42)
    >>> buf.push(state, action=0, reward=1.0, next_state=next_s, done=False)
    >>> if buf.is_ready(batch_size=64):
    ...     states, actions, rewards, next_states, dones = buf.sample(64, device)
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a single transition.

        All inputs are stored as-is. No encoding, normalization,
        or transformation is applied.

        Parameters
        ----------
        state : np.ndarray
            Encoded state vector, shape ``(state_dim,)``, dtype float32.
        action : int
            Discrete action index selected by the agent.
        reward : float
            Scalar reward received from the environment.
        next_state : np.ndarray
            Encoded next-state vector, shape ``(state_dim,)``, dtype float32.
        done : bool
            Whether the episode terminated after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[
        torch.FloatTensor,
        torch.LongTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """
        Sample a uniformly random mini-batch and return as tensors.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        device : torch.device
            Target device for all returned tensors.

        Returns
        -------
        states : torch.FloatTensor
            Shape ``(batch_size, state_dim)``.
        actions : torch.LongTensor
            Shape ``(batch_size,)``.
        rewards : torch.FloatTensor
            Shape ``(batch_size,)``.
        next_states : torch.FloatTensor
            Shape ``(batch_size, state_dim)``.
        dones : torch.FloatTensor
            Shape ``(batch_size,)``. Values are ``0.0`` or ``1.0``.

        Raises
        ------
        ValueError
            If ``batch_size`` exceeds current buffer size.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {len(self.buffer)}"
            )

        batch = self.rng.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones, dtype=np.float32)).to(device),
        )

    def is_ready(self, batch_size: int) -> bool:
        """
        Check whether buffer has enough transitions to sample.

        Parameters
        ----------
        batch_size : int
            Desired batch size.

        Returns
        -------
        bool
            True if current size >= batch_size.
        """
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return len(self.buffer)