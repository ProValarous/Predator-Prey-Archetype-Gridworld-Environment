"""
Reward registry.

Maps string identifiers (from YAML configs) to RewardFunction classes.
"""

from typing import Type

from ppage.rewards.base import RewardFunction
from ppage.rewards.base_reward import BaseReward
from ppage.rewards.predator_distance import PredatorDistanceReward
from ppage.rewards.survival_reward import SurvivalReward

_REWARD_REGISTRY: dict[str, Type[RewardFunction]] = {
    "base": BaseReward,
    "predator_distance": PredatorDistanceReward,
    "survival": SurvivalReward,
}


def get_reward_function(name: str, weight: float = 1.0, **params) -> RewardFunction:
    """
    Instantiate a reward function by name.

    Args:
        name: registry key (string)
        weight: scalar multiplier
        params: additional YAML parameters

    Returns:
        RewardFunction instance
    """
    if name not in _REWARD_REGISTRY:
        raise KeyError(
            f"Unknown reward type '{name}'. "
            f"Available: {list(_REWARD_REGISTRY.keys())}"
        )

    return _REWARD_REGISTRY[name](weight=weight, **params)


def register_reward(name: str, cls: Type[RewardFunction]) -> None:
    """
    Register a new reward function.

    Intended for advanced users only.
    """
    if not issubclass(cls, RewardFunction):
        raise TypeError("Reward must inherit from RewardFunction")

    _REWARD_REGISTRY[name] = cls
