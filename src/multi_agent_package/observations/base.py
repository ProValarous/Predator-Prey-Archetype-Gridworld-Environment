"""
Base observation interface.

All custom observation builders MUST inherit from ObservationBuilder
and implement the build(env) method.

Rules:
- Do NOT modify env state
- Do NOT move agents
- Use only public agent attributes and positions
"""

from abc import ABC, abstractmethod
from typing import Dict


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
