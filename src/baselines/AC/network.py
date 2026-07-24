# src/baselines/AC/network.py
"""
Configurable actor-critic network (PyTorch).

Shared trunk feeds two heads: a policy head producing per-action logits
(consumed as a Categorical distribution) and a value head producing a
scalar state-value estimate. Same trunk-then-heads shape as
DuelingQNetwork, repurposed for policy-gradient learning instead of
Q-value decomposition. Output layers are linear -- logits must be free
to take any sign, and the value estimate is unbounded.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], action_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if not hidden_layers or any(h <= 0 for h in hidden_layers):
            raise ValueError(
                "hidden_layers must be a non-empty list of positive ints, "
                f"got {hidden_layers}"
            )

        trunk: List[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            trunk.append(nn.Linear(in_dim, int(hidden_dim)))
            trunk.append(nn.ReLU())
            in_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*trunk)

        self.policy_head = nn.Linear(in_dim, int(action_dim))
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        action_logits = self.policy_head(features)
        state_value = self.value_head(features).squeeze(-1)
        return action_logits, state_value
