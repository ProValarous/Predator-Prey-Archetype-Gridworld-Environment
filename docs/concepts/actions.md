# Concept: Action Spaces

## What an Action Space Is

An **action space** defines the set of moves an agent can make and translates discrete integer commands into grid movement vectors. Every call to `env.step(actions)` passes integer actions; the action space plugin converts each integer to a `[dx, dy]` direction vector before movement is applied.

Implemented as a plugin in `src/multi_agent_package/actions/`.

---

## Shipped Action Spaces

Three action spaces are registered in `action_registry.py`.

### `discrete_5` — `DiscreteActionSpace` (default)

| Index | Label | Direction Vector | Notes |
|-------|-------|-----------------|-------|
| 0 | RIGHT | `[+1,  0]` | |
| 1 | UP    | `[ 0, +1]` | Y increases downward — visually moves **down** on screen |
| 2 | LEFT  | `[-1,  0]` | |
| 3 | DOWN  | `[ 0, -1]` | Y decreases — visually moves **up** on screen |
| 4 | NOOP  | `[ 0,  0]` | Stay in place |

**Coordinate system caveat:** Origin `[0,0]` is at the top-left. Y increases downward. Action 1 labeled "Up" moves toward higher Y, which appears visually downward on screen. Reason from the direction vectors, not the labels.

### `cross` — `CrossActionSpace`

Diagonal-only movement — no cardinal (up/down/left/right) moves at all, despite what "cross" might suggest. It's an independent 5-action space (subclasses `ActionSpace` directly, not `DiscreteActionSpace`):

| Index | Label | Direction Vector | Notes |
|-------|-------|-----------------|-------|
| 0 | NE | `[+1, +1]` | Visually down-right (Y increases downward) |
| 1 | NW | `[-1, +1]` | Visually down-left |
| 2 | SW | `[-1, -1]` | Visually up-left |
| 3 | SE | `[+1, -1]` | Visually up-right |
| 4 | NOOP | `[0, 0]` | Stay in place |

Same coordinate-system caveat as `discrete_5`: reason from the vectors, not the compass labels.

### `speed_discrete_5` — `SpeedDiscreteActionSpace`

Subclasses `DiscreteActionSpace` and inherits its exact 5-action mapping unchanged — "speed" is not extra action indices. It adds one helper, `to_moves(action, speed, stamina) -> list`, which returns up to `min(speed, stamina)` copies of the move vector (empty for NOOP). It is a standalone utility; the [SpeedWrapper](wrappers.md) itself no longer calls it (the wrapper reads `is_noop` from the configured action plugin instead):

```python
def to_moves(self, action, speed, stamina) -> list:
    direction = self.to_direction(action)
    if not direction.any():          # NOOP — no movement, no stamina cost
        return []
    n = min(max(speed, 1), max(stamina, 0))
    return [direction] * n           # n identical direction vectors
```

`to_moves()` returns real direction vectors, but nothing in the shipped code path
consumes them — the `SpeedWrapper` now computes its sub-step budget from
`plugin.is_noop(...)` and `min(speed, stamina)` directly, and re-sends the original
action index for each sub-step. `to_moves()` remains available for custom code.

---

## Why a Plugin

Action spaces were originally hardcoded in `Agent._action_to_direction()`. Extracting them into a plugin follows the same pattern as observations and rewards:

- A new action space (e.g. 8-directional movement) is a new file — no changes to core
- The active action space is declared in `configs/actions.yaml`
- Baselines can inspect `env.action_space_plugin.gymnasium_space` and `n_actions` instead of hard-coding `Discrete(5)` (DQN does this via `_resolve_action_dim`; IQL/CQL/MixedTrainer still take a fixed `action_dim` from config instead)

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

> **Registry validation:** `register_action_space()` raises `TypeError` if the
> class is not an `ActionSpace` subclass, consistent with `register_reward()` and
> `register_observation()`. All three registries reject a bad class at
> registration time rather than failing later inside `env.step()`.
