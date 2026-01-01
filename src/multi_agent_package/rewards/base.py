"""
Base reward interface.

All custom reward functions MUST inherit from RewardFunction
and implement the compute(env) method.

Rules:
- Rewards are functions of state, NOT actions
- Do NOT modify env or agent state
- Do NOT terminate episodes
"""

from abc import ABC, abstractmethod
from typing import Dict


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    """

    def __init__(self, weight: float = 1.0, **kwargs):
        """
        Args:
            weight: scalar multiplier applied to this reward
            kwargs: optional config parameters
        """
        self.weight = float(weight)
        self.params = kwargs

    @abstractmethod
    def compute(self, env) -> Dict[str, float]:
        """
        Compute rewards for all agents.

        Args:
            env: GridWorldEnv (READ-ONLY)

        Returns:
            Dict[agent_name, reward_value]
        """
        raise NotImplementedError
