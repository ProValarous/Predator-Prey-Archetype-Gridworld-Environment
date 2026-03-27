"""
DQN Network Definitions.

This module defines the neural network architecture for DQN.
It contains NO training logic, NO environment interaction,
and NO agent decision logic.

Two modes controlled by use_dueling flag:

    use_dueling=True  (default, recommended)
        Dueling DQN architecture.
        Network splits into two heads after shared layers:
            V(s)    : scalar   — how good is this state?
            A(s, a) : vector   — how much better is each action vs average?
        Combined as: Q(s, a) = V(s) + A(s, a) - mean_a(A(s, a))
        Benefit: learns state value independently from action advantage.
        Better at identifying states where actions don't matter much.

    use_dueling=False
        Vanilla / Double DQN architecture.
        Single head outputs Q(s, a) directly for all actions.

Network input:  flat float32 vector  shape (state_dim,)
Network output: Q-values             shape (action_dim,)

Architecture:
    Input (state_dim)
        → Linear(state_dim, hidden[0]) + ReLU
        → Linear(hidden[i], hidden[i+1]) + ReLU  (for each pair)
        → [dueling]  V-head: Linear(hidden[-1], 1)
                     A-head: Linear(hidden[-1], action_dim)
          [vanilla]  Q-head: Linear(hidden[-1], action_dim)
"""

from typing import List

import torch
import torch.nn as nn


class DuelingMLP(nn.Module):
    """
    Fully-connected network with optional dueling heads.

    Supports both Dueling DQN and vanilla/Double DQN architectures
    via the use_dueling flag. The external interface is identical in
    both modes: input a state vector, get Q-values out.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the flattened input state vector.
    action_dim : int
        Number of discrete actions.
    hidden_units : list of int
        Sizes of hidden layers. e.g. [128, 128] gives two hidden
        layers of 128 units each. Must have at least one element.
    use_dueling : bool, optional
        If True, splits final layer into V and A heads (default True).

    Examples
    --------
    >>> net = DuelingMLP(state_dim=10, action_dim=5, hidden_units=[128, 128])
    >>> obs = torch.zeros(1, 10)   # batch of 1
    >>> q_values = net(obs)        # shape (1, 5)

    >>> net_vanilla = DuelingMLP(state_dim=10, action_dim=5,
    ...                          hidden_units=[64], use_dueling=False)
    >>> q_values = net_vanilla(obs)  # shape (1, 5) — same interface
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: List[int],
        use_dueling: bool = True,
    ) -> None:
        super().__init__()

        if len(hidden_units) == 0:
            raise ValueError("hidden_units must have at least one element.")

        self.use_dueling = use_dueling

        # ── Shared feature layers ─────────────────────────────────────────
        # Build: input → hidden[0] → hidden[1] → ... → hidden[-1]
        layers = []
        in_dim = state_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.shared = nn.Sequential(*layers)

        # ── Output heads ─────────────────────────────────────────────────
        if self.use_dueling:
            # Value head: estimates V(s) — scalar per state
            self.value_head = nn.Linear(in_dim, 1)
            # Advantage head: estimates A(s,a) — one value per action
            self.advantage_head = nn.Linear(in_dim, action_dim)
        else:
            # Single Q head: estimates Q(s,a) directly
            self.q_head = nn.Linear(in_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for a batch of states.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, state_dim)``, dtype float32.

        Returns
        -------
        torch.Tensor
            Q-values, shape ``(batch_size, action_dim)``, dtype float32.
        """
        features = self.shared(x)

        if self.use_dueling:
            v = self.value_head(features)           # (batch, 1)
            a = self.advantage_head(features)       # (batch, action_dim)

            # Combine: Q = V + (A - mean(A))
            # Subtracting mean(A) makes the decomposition unique —
            # without it V and A are unidentifiable (many V,A pairs
            # give the same Q). keepdim=True keeps shape (batch, 1)
            # so the subtraction broadcasts correctly across actions.
            q = v + (a - a.mean(dim=1, keepdim=True))
        else:
            q = self.q_head(features)               # (batch, action_dim)

        return q