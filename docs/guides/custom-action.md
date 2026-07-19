# Guide: Writing a Custom Action Space

Full working example — from file to config-driven.

---

## Step 1 — Create the file

```
src/multi_agent_package/actions/my_actions.py
```

Example: an 8-directional action space adding diagonal moves.

```python
import numpy as np
from gymnasium import spaces
from multi_agent_package.actions.base import ActionSpace


class EightDirectionalActionSpace(ActionSpace):
    """
    Eight-action space: cardinal + diagonal movement, plus noop.

    Actions:
        0: RIGHT       → [+1,  0]
        1: UP          → [ 0, +1]
        2: LEFT        → [-1,  0]
        3: DOWN        → [ 0, -1]
        4: UP-RIGHT    → [+1, +1]
        5: UP-LEFT     → [-1, +1]
        6: DOWN-RIGHT  → [+1, -1]
        7: DOWN-LEFT   → [-1, -1]
        8: NOOP        → [ 0,  0]
    """

    _DIRECTIONS = {
        0: np.array([ 1,  0], dtype=np.int32),
        1: np.array([ 0,  1], dtype=np.int32),
        2: np.array([-1,  0], dtype=np.int32),
        3: np.array([ 0, -1], dtype=np.int32),
        4: np.array([ 1,  1], dtype=np.int32),
        5: np.array([-1,  1], dtype=np.int32),
        6: np.array([ 1, -1], dtype=np.int32),
        7: np.array([-1, -1], dtype=np.int32),
        8: np.array([ 0,  0], dtype=np.int32),
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
```

**Rules:**
- `to_direction()` must raise `ValueError` for unknown action integers — never silently return zero
- Direction vectors must be `np.int32` arrays of shape `(2,)`
- `gymnasium_space.n` must equal `n_actions`
- Never access `env` or agent state inside `to_direction()`

---

## Step 2 — Register it

Open `src/multi_agent_package/registry/action_registry.py`:

```python
from multi_agent_package.actions.my_actions import EightDirectionalActionSpace

_ACTION_REGISTRY = {
    "discrete_5":   DiscreteActionSpace,
    "discrete_9":   EightDirectionalActionSpace,   # ← add this line
}
```

---

## Step 3 — Export it

Open `src/multi_agent_package/actions/__init__.py`:

```python
from .my_actions import EightDirectionalActionSpace

__all__ = [
    "ActionSpace",
    "DiscreteActionSpace",
    "EightDirectionalActionSpace",   # ← add this line
]
```

---

## Step 4 — Configure it

```yaml
# configs/actions.yaml
actions:
  type: discrete_9   # ← matches your registry key
  params: {}
```

---

## Step 5 — Update baselines (if needed)

Existing baselines (IQL, CQL) hard-code `action_dim: 5` in `experiment.yaml`. If your action space has a different number of actions, update that param:

```yaml
# configs/experiment.yaml
experiment:
  algorithm:
    params:
      action_dim: 9   # ← must match your action space's n_actions
```

Alternatively, baselines can read `env.action_space_plugin.n_actions` directly to avoid the mismatch:
```python
action_dim = env.action_space_plugin.n_actions if env.action_space_plugin else 5
```

---

## Step 6 — Test it

```python
from multi_agent_package.actions.my_actions import EightDirectionalActionSpace
import numpy as np

space = EightDirectionalActionSpace()

assert space.n_actions == 9
assert space.gymnasium_space.n == 9

# Valid action
d = space.to_direction(4)   # UP-RIGHT
assert d.tolist() == [1, 1]

# Invalid action raises
try:
    space.to_direction(99)
    assert False, "should have raised"
except ValueError:
    pass

# Wire to environment
from multi_agent_package.core.gridworld import GridWorldEnv
from multi_agent_package.core.agent import Agent

agents = [
    Agent(agent_type="predator", agent_team="predator_1", agent_name="predator_1"),
    Agent(agent_type="prey",     agent_team="prey_1",     agent_name="prey_1"),
]
env = GridWorldEnv(agents=agents, size=8, perc_num_obstacle=0, render_mode=None, seed=0)
env.action_space_plugin = space
env.reset()

result = env.step({"predator_1": 4, "prey_1": 8})   # predator moves UP-RIGHT, prey stays
assert "predator_1" in result["obs"]

print("All assertions passed.")
```

---

## Checklist

- [ ] Inherits from `ActionSpace`
- [ ] `to_direction()` raises `ValueError` for out-of-range integers
- [ ] `gymnasium_space.n == n_actions`
- [ ] Direction vectors are `np.int32`, shape `(2,)`
- [ ] Registered in `action_registry.py` with matching YAML key
- [ ] Exported from `actions/__init__.py`
- [ ] `action_dim` updated in `experiment.yaml` (or read from plugin)
- [ ] Added entry to [concepts/actions.md](../concepts/actions.md)
