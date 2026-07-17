# Spec: ObservationBuilder

Formal contract for all observation builder implementations.

---

## Identity

| Property | Value |
|----------|-------|
| Abstract base | `ObservationBuilder` |
| File | `src/multi_agent_package/observations/base.py` |
| Registry | `observation_registry.py` |
| Extensible by | All contributors |

---

## Required Interface

```python
class MyObservation(ObservationBuilder):
    def __init__(self, **params):
        super().__init__(**params)
        # extract what you need from params

    def build(self, env) -> Dict[str, dict]:
        # must return one entry per agent in env.agents
        ...
```

---

## `build(env)` Contract

### Preconditions
- `env.agents` is non-empty
- `env.reset()` has been called (agent positions are valid)

### Postconditions
- Returns a dict with exactly one key per agent name in `env.agents`
- Each value is a dict (structure is builder-specific but must be consistent across calls)
- `env` state is **unchanged** — no attribute may be written
- Agent state is **unchanged** — `agent._agent_location` must not be written

### Determinism requirement
Given the same `env` state (positions, captured set), `build(env)` must return the same output. No stochastic elements, no calls to `random`, no time-dependent values.

---

## What `build()` May Access

| Attribute | Allowed? |
|-----------|----------|
| `env.agents` (read) | ✅ |
| `env._obstacle_location` (read) | ✅ |
| `env.size` (read) | ✅ |
| `env._captured_agents` (read) | ✅ |
| `agent._agent_location` (read) | ✅ |
| `agent.agent_type` (read) | ✅ |
| `agent.agent_name` (read) | ✅ |
| `env._default_observations()` (call) | ✅ (DefaultObservation only) |
| `env._obstacle_location` (write) | ❌ |
| `agent._agent_location` (write) | ❌ |
| `env.reward_fn` (call) | ❌ |
| `env.step()` | ❌ |

---

## Registration

Every builder must be registered before it can be used via config:

```python
# In observation_registry.py — add to _OBSERVATION_REGISTRY:
"my_obs": MyObservation,

# Or at runtime (e.g. in a custom script):
from multi_agent_package.registry.observation_registry import register_observation
register_observation("my_obs", MyObservation)
```

The registry key must match what you put in `observations.yaml`:
```yaml
observations:
  type: my_obs
  params:
    my_param: value
```

---

## Output Structure Requirements

The output structure is **builder-specific** but must be:
1. Consistent across calls (same keys every step for the same builder)
2. Serializable to a hashable tuple by `IQL._encode_state()` (no non-serializable objects)
3. Dict-of-dicts at the top level: `{agent_name: {field: value, ...}}`

Recommended minimum structure:
```python
{
    "agent_name": {
        "local": np.ndarray,  # agent's own position [x, y]
        # ... builder-specific fields
    }
}
```

---

## Checklist for New Builders

Before submitting a new observation builder:

- [ ] Inherits from `ObservationBuilder`
- [ ] `build(env)` covers all agents in `env.agents`
- [ ] No writes to env or agent state inside `build()`
- [ ] `__init__` uses only kwargs (no positional args beyond `self`)
- [ ] Registered in `observation_registry.py` with correct key (no typos)
- [ ] Exported from `observations/__init__.py`
- [ ] Added an entry to [concepts/observations.md](../concepts/observations.md)
