# Concept: Action Spaces

## What an Action Space Is

An **action space** defines the set of moves an agent can make and translates discrete integer commands into grid movement vectors. Every call to `env.step(actions)` passes integer actions; the action space plugin converts each integer to a `[dx, dy]` direction vector before movement is applied.

Implemented as a plugin in `src/multi_agent_package/actions/`.

---

## The Current Action Space

`DiscreteActionSpace` — five-action discrete space, registered as `"discrete_5"`.

| Index | Label | Direction Vector | Notes |
|-------|-------|-----------------|-------|
| 0 | RIGHT | `[+1,  0]` | |
| 1 | UP    | `[ 0, +1]` | Y increases downward — visually moves **down** on screen |
| 2 | LEFT  | `[-1,  0]` | |
| 3 | DOWN  | `[ 0, -1]` | Y decreases — visually moves **up** on screen |
| 4 | NOOP  | `[ 0,  0]` | Stay in place |

**Coordinate system caveat:** Origin `[0,0]` is at the top-left. Y increases downward. Action 1 labeled "Up" moves toward higher Y, which appears visually downward on screen. Reason from the direction vectors, not the labels.

---

## Why a Plugin

Action spaces were originally hardcoded in `Agent._action_to_direction()`. Extracting them into a plugin follows the same pattern as observations and rewards:

- A new action space (e.g. 8-directional movement, diagonal-only) is a new file — no changes to core
- The active action space is declared in `configs/actions.yaml`
- Baselines can inspect `env.action_space_plugin.gymnasium_space` and `n_actions` instead of hard-coding `Discrete(5)`

See [ADR-005](../decisions/ADR-005-action-plugin.md) for the full rationale.

---

## Plugin Interface

```python
class ActionSpace(ABC):
    def to_direction(self, action: int) -> np.ndarray:
        """Map action integer to [dx, dy] movement vector."""

    @property
    def gymnasium_space(self) -> spaces.Space:
        """Return the corresponding gymnasium action space."""

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
```

---

## Backward Compatibility

`Agent._actions_to_directions` still exists and still holds the same mapping. If `env.action_space_plugin` is `None`, `step()` falls back to the agent's own dict. This means existing code that does not wire the plugin continues to work unchanged.

---

## Relationship to Other Plugins

| Plugin | Answers the question | Called in |
|--------|---------------------|-----------|
| `ActionSpace` | What can an agent do? | `env.step()` — movement phase |
| `ObservationBuilder` | What does an agent see? | `env.step()` — observations phase |
| `RewardFunction` | What does an agent earn? | `env.step()` — rewards phase |

All three are wired in `run_from_config.build_environment()` and injected as hooks on the env.
