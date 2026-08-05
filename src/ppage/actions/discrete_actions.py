"""
Standard 5-action discrete action space.

Actions:
    0: RIGHT  → [+1,  0]
    1: UP     → [ 0, +1]
    2: LEFT   → [-1,  0]
    3: DOWN   → [ 0, -1]
    4: NOOP   → [ 0,  0]
"""

import numpy as np
from gymnasium import spaces

from .base import ActionSpace


class DiscreteActionSpace(ActionSpace):
    """
    Five-action discrete space: RIGHT, UP, LEFT, DOWN, NOOP.
    """

    _DIRECTIONS = {
        0: np.array([1, 0], dtype=np.int32),
        1: np.array([0, 1], dtype=np.int32),
        2: np.array([-1, 0], dtype=np.int32),
        3: np.array([0, -1], dtype=np.int32),
        4: np.array([0, 0], dtype=np.int32),
    }

    def to_direction(self, action: int) -> np.ndarray:
        if action not in self._DIRECTIONS:
            raise ValueError(
                f"Invalid action {action!r}. Must be one of {list(self._DIRECTIONS)}."
            )
        return self._DIRECTIONS[action]

    @property
    def gymnasium_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self._DIRECTIONS))

    @property
    def n_actions(self) -> int:
        return len(self._DIRECTIONS)
