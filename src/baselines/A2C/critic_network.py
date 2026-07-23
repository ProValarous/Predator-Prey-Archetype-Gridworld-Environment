# src/baselines/A2C/critic_network.py
"""
Configurable feedforward state-value (critic) network (PyTorch).

Estimates V(s), a single scalar per state. Output layer is linear 
(handle negative penalties)
"""

from typing import List

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int]):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if not hidden_layers or any(h <= 0 for h in hidden_layers):
            raise ValueError(
                f"hidden_layers must be a non-empty list of positive ints, got {hidden_layers}"
            )

        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """V(s), shape (batch, 1)."""
        return self.model(x)
