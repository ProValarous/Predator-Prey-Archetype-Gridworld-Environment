from typing import Dict, Type

from multi_agent_package.actions.base import ActionSpace
from multi_agent_package.actions.discrete_actions import DiscreteActionSpace
from multi_agent_package.actions.cross_actions import CrossActionSpace
from multi_agent_package.actions.speed_discrete import SpeedDiscreteActionSpace

_ACTION_REGISTRY: Dict[str, Type[ActionSpace]] = {
    "discrete_5": DiscreteActionSpace,
    "cross": CrossActionSpace,
    "speed_discrete_5": SpeedDiscreteActionSpace,
}


def get_action_space(name: str, **params) -> ActionSpace:
    if name not in _ACTION_REGISTRY:
        raise KeyError(
            f"Unknown action space '{name}'. " f"Available: {list(_ACTION_REGISTRY)}"
        )
    return _ACTION_REGISTRY[name](**params)


def register_action_space(name: str, cls: Type[ActionSpace]) -> None:
    _ACTION_REGISTRY[name] = cls
