"""
Observation registry.

Maps string identifiers (from YAML configs) to ObservationBuilder classes.

Students MUST register new observation builders here.
"""

from typing import Type

from multi_agent_package.observations.base import ObservationBuilder
from multi_agent_package.observations.default import DefaultObservation
from multi_agent_package.observations.local_only import LocalOnlyObservation
from multi_agent_package.observations.local_radius import LocalRadiusObservation


_OBSERVATION_REGISTRY: dict[str, Type[ObservationBuilder]] = {
    "default": DefaultObservation,
    "local_only": LocalOnlyObservation,
    "local_radius": LocalRadiusObservation,
}


def get_observation_builder(name: str, **params) -> ObservationBuilder:
    """
    Instantiate an observation builder by name.

    Args:
        name: registry key (string)
        params: keyword arguments passed from YAML

    Returns:
        ObservationBuilder instance
    """
    if name not in _OBSERVATION_REGISTRY:
        raise KeyError(
            f"Unknown observation type '{name}'. "
            f"Available: {list(_OBSERVATION_REGISTRY.keys())}"
        )

    return _OBSERVATION_REGISTRY[name](**params)


def register_observation(name: str, cls: Type[ObservationBuilder]) -> None:
    """
    Register a new observation builder.

    Intended for advanced users only.
    """
    if not issubclass(cls, ObservationBuilder):
        raise TypeError("Observation must inherit from ObservationBuilder")

    _OBSERVATION_REGISTRY[name] = cls
