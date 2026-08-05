"""
Base observation interface.

All custom observation builders MUST inherit from ObservationBuilder,
implement build(env), and provide an encode(observation, env) method for
tensor-facing consumers such as DQN.

Rules:
- Do NOT modify env state
- Do NOT move agents
- Use only public agent attributes and positions
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class ObservationBuilder(ABC):
    """
    Abstract base class for observation builders.
    """

    def __init__(self, **kwargs):
        """
        Parameters are passed from YAML config.
        """
        self.params = kwargs

    @abstractmethod
    def build(self, env) -> Dict[str, dict]:
        """
        Build observations for all agents.

        Args:
            env: GridWorldEnv (read-only usage)

        Returns:
            Dict[agent_name, observation_dict]
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, observation: dict, env) -> np.ndarray:
        """
        Convert a single agent observation into a fixed numeric vector.

        Args:
            observation: one agent's observation payload
            env: GridWorldEnv instance used for padding / shape context

        Returns:
            1D float32 numpy array
        """
        raise NotImplementedError

    @staticmethod
    def _agent_type_id(agent_type) -> float:
        label = str(agent_type).lower()
        if label.startswith("predator"):
            return 0.0
        if label.startswith("prey"):
            return 1.0
        return 2.0

    @staticmethod
    def _team_features(team) -> Tuple[float, float]:
        if team is None:
            return 2.0, 0.0

        if isinstance(team, int):
            return 2.0, float(team)

        label = str(team).lower()
        if "_" in label:
            base, suffix = label.split("_", 1)
            try:
                return ObservationBuilder._agent_type_id(base), float(int(suffix))
            except ValueError:
                return ObservationBuilder._agent_type_id(base), 1.0

        if label.isdigit():
            return 2.0, float(int(label))

        return ObservationBuilder._agent_type_id(label), 1.0

    @staticmethod
    def _vector(values) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape(-1)
