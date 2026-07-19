# src/baselines/DQN/q_network.py
"""
Configurable feedforward Q-network (PyTorch).

hidden_layers is a list of hidden-layer widths, e.g. [128, 128] = two
hidden layers of 128 units each, [64, 32, 16] = three layers. Output
layer is linear (no activation) since Q-values must be free to go
negative.
"""

from typing import List

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not hidden_layers or any(h <= 0 for h in hidden_layers):
            raise ValueError(
                "hidden_layers must be a non-empty list of positive ints, "
                f"got {hidden_layers}"
            )

        layers: List[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
