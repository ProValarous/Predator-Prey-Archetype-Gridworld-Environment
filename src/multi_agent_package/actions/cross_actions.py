"""
Diagonal ("cross") discrete action space.

Agents using this action space can move ONLY along diagonals — no
cardinal (up/down/left/right) movement is possible. This is a sibling
of DiscreteActionSpace, not a modification of it.

Actions:
    0: NE (+x, +y)  → [+1, +1]
    1: NW (-x, +y)  → [-1, +1]
    2: SW (-x, -y)  → [-1, -1]
    3: SE (+x, -y)  → [+1, -1]
    4: NOOP         → [ 0,  0]

Coordinate note: origin [0,0] is top-left and Y increases downward,
same convention as DiscreteActionSpace. So action 0 ([+1,+1]) moves the
agent one cell right and one cell toward higher Y — visually
down-right on screen. As with the cardinal space, reason from the
direction vectors, not the compass labels.

it inherits from ActionSpace directly (same as DiscreteActionSpace does), 
not from DiscreteActionSpace — these are two independent 5-action spaces, 
not a specialization of one another.
"""

import numpy as np
from gymnasium import spaces

from .base import ActionSpace


class CrossActionSpace(ActionSpace):
    """
    Five-action discrete space: four diagonal moves + NOOP.
    No cardinal (straight) movement exists in this action space.
    """

    _DIRECTIONS = {
        0: np.array([1, 1], dtype=np.int32),    # NE
        1: np.array([-1, 1], dtype=np.int32),   # NW
        2: np.array([-1, -1], dtype=np.int32),  # SW
        3: np.array([1, -1], dtype=np.int32),   # SE
        4: np.array([0, 0], dtype=np.int32),    # NOOP
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