# src/baselines/A2C/actor_network.py
"""
Configurable feedforward policy (actor) network for discrete action
spaces (PyTorch).

input_dim -> hidden_layers -> output_dim),
the output layer here is interpreted
as unnormalized action LOGITS

Wrapping the logits in torch.distributions.Categorical gives everything
A2C needs from one forward pass:
    dist.sample()    -> stochastic action (exploration)
    dist.log_prob(a) -> log pi(a|s), used in the policy-gradient loss
    dist.entropy()   -> entropy bonus, keeps the policy from collapsing
                        to a single action too early
"""

from typing import List

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
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
        # Linear output -- these are LOGITS, not probabilities. Categorical
        # applies softmax internally
        layers.append(nn.Linear(in_dim, int(output_dim)))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw action logits, shape (batch, output_dim)."""
        return self.model(x)

    def distribution(self, x: torch.Tensor) -> Categorical:
        """Wrap forward()'s logits in a Categorical distribution."""
        logits = self.forward(x)
        return Categorical(logits=logits)
