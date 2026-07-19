# Spec: ActionSpace

Formal contract for all action space implementations.

---

## Identity

| Property | Value |
|----------|-------|
| Abstract base | `ActionSpace` |
| File | `src/multi_agent_package/actions/base.py` |
| Registry | `action_registry.py` |
| Extensible by | All contributors |

---

## Required Interface

```python
class MyActionSpace(ActionSpace):
    def __init__(self, **params):
        super().__init__(**params)

    def to_direction(self, action: int) -> np.ndarray:
        # return [dx, dy] for the given action integer
        ...

    @property
    def gymnasium_space(self) -> spaces.Space:
        # return the gymnasium space matching this action set
        ...

    @property
    def n_actions(self) -> int:
        # number of valid action integers (0 to n_actions-1)
        ...
```

---

## `to_direction(action)` Contract

### Preconditions
- `action` is an integer in the range `[0, n_actions - 1]`

### Postconditions
- Returns an `np.ndarray` of shape `(2,)` and dtype compatible with `int32`
- The vector represents `[dx, dy]` in grid coordinates (origin top-left, Y increases downward)
- For the same `action` input, always returns the same vector (deterministic)
- Raises `ValueError` for out-of-range action integers (do not silently return zero)

### What `to_direction()` must NOT do
- Modify `env` state
- Access `env` at all — direction lookup must be a pure function of the action integer
- Return vectors with magnitude > 1 (single-step movement only)

---

## `gymnasium_space` Contract

- Must return a `gymnasium.spaces.Space`
- Must be consistent with `n_actions`: `gymnasium_space.n == n_actions` for `Discrete` spaces
- Must be stable — same object or equivalent across calls

---

## `n_actions` Contract

- Returns the count of valid action integers
- Valid integers are `0, 1, ..., n_actions - 1`

---

## What `to_direction()` May Access

| Resource | Allowed? |
|----------|----------|
| `self.params` (read) | ✅ |
| Instance attributes (read) | ✅ |
| `env` or `agent` state | ❌ |
| External randomness | ❌ |

---

## Registration

Every action space must be registered before it can be used via config:

```python
# In action_registry.py — add to _ACTION_REGISTRY:
"my_action_space": MyActionSpace,

# Or at runtime:
from multi_agent_package.registry.action_registry import register_action_space
register_action_space("my_action_space", MyActionSpace)
```

> ⚠️ Unlike `register_reward()` and `register_observation()`, `register_action_space()` does **not** check `issubclass(cls, ActionSpace)` — registering an incompatible class succeeds silently and only fails later, when `get_action_space()`'s result is actually used inside `env.step()`.

The registry key must match what is in `actions.yaml`:
```yaml
actions:
  type: my_action_space
  params:
    my_param: value
```

---

## Wiring

The action space is set on the environment before training begins:

```python
env.action_space_plugin = get_action_space("discrete_5")
```

`step()` calls `env.action_space_plugin.to_direction(a)` when the plugin is set, and falls back to `ag._actions_to_directions[a]` otherwise.

> **`SpeedWrapper` uses `env.action_space_plugin`.** It reads NOOP-ness via
> `plugin.is_noop(action)` and derives the idle-slot action index from the same
> configured plugin, so a custom action space with a different NOOP index works
> correctly. (`to_moves()` on `SpeedDiscreteActionSpace` still exists but the
> wrapper no longer uses it.) See [Wrappers](../concepts/wrappers.md).

---

## Checklist for New Action Spaces

Before submitting a new action space:

- [ ] Inherits from `ActionSpace`
- [ ] `to_direction()` raises `ValueError` for invalid action integers
- [ ] `gymnasium_space.n == n_actions`
- [ ] Direction vectors are `np.ndarray` of shape `(2,)`, dtype compatible with `int32`
- [ ] All direction vectors have component values in `{-1, 0, +1}` (single-step)
- [ ] `__init__` uses only `**kwargs` (no positional args beyond `self`)
- [ ] Registered in `action_registry.py` with correct key
- [ ] Exported from `actions/__init__.py`
- [ ] Added an entry to [concepts/actions.md](../concepts/actions.md)
