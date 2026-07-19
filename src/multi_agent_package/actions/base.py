"""
Base action space interface.

All custom action spaces MUST inherit from ActionSpace
and implement to_direction() and the gymnasium_space / n_actions properties.

Rules:
- Do NOT modify env state
- to_direction() must return a valid numpy [dx, dy] vector
- gymnasium_space must be consistent with the set of valid action integers
"""

from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces


class ActionSpace(ABC):
    """
    Abstract base class for action spaces.
    """

    def __init__(self, **kwargs):
        """
        Parameters are passed from YAML config.
        """
        self.params = kwargs

    @abstractmethod
    def to_direction(self, action: int) -> np.ndarray:
        """
        Map a discrete action integer to a movement direction vector.

        Args:
            action: integer action index

        Returns:
            np.ndarray of shape (2,) representing [dx, dy]
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def gymnasium_space(self) -> spaces.Space:
        """
        Return the corresponding gymnasium action space.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """
        Number of discrete actions.
        """
        raise NotImplementedError
