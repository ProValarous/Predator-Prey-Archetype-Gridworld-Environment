# Spec: RewardFunction

Formal contract for all reward function implementations.

---

## Identity

| Property | Value |
|----------|-------|
| Abstract base | `RewardFunction` |
| File | `src/multi_agent_package/rewards/base.py` |
| Registry | `reward_registry.py` |
| Extensible by | All contributors |

---

## Required Interface

```python
class MyReward(RewardFunction):
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(weight=weight, **kwargs)
        # extract custom params from kwargs

    def compute(self, env) -> Dict[str, float]:
        # return per-agent reward contribution
        ...
```

---

## `compute(env)` Contract

### Preconditions
- `env.reset()` has been called
- Capture detection for the current step has already run (env captures are up-to-date)

### Postconditions
- Returns a dict mapping agent names to float reward values
- Missing agent names default to `0.0` in the composition closure
- `env` state is **unchanged** — no attribute may be written
- Agent state is **unchanged**
- Termination conditions are **not modified** (reward functions cannot end episodes)

### Determinism requirement
Given the same `env` state, `compute(env)` must return the same output. No randomness.

---

## What `compute()` May Access

| Attribute | Allowed? |
|-----------|----------|
| `env.agents` (read) | ✅ |
| `env._obstacle_location` (read) | ✅ |
| `env._captured_agents` (read) | ✅ |
| `env._captures_total` (read) | ✅ |
| `env.base_reward()` (call) | ✅ (BaseReward only) |
| `agent._agent_location` (read) | ✅ |
| `agent.agent_type` (read) | ✅ |
| `env._captured_agents` (write) | ❌ |
| `agent._agent_location` (write) | ❌ |
| `env.observation_builder` (call) | ❌ |
| `env.step()` | ❌ |

---

## `weight` Semantics

The `weight` parameter is a scalar multiplier applied inside `compute()`. Convention:

```python
def compute(self, env):
    rewards = {}
    for agent in env.agents:
        raw = ...  # compute raw signal
        rewards[agent.agent_name] = raw * self.weight
    return rewards
```

`weight=0` effectively disables a reward function without removing it from the composition.

---

## Composition Behavior

Reward functions are summed per agent by the `combined_reward` closure. Each function is responsible only for its own contribution — do not account for other reward functions inside your `compute()`.

```
total[agent] = fn1.compute(env)[agent]
             + fn2.compute(env)[agent]
             + fn3.compute(env)[agent]
             + ...
```

---

## Registration

```python
# In reward_registry.py — add to _REWARD_REGISTRY:
"my_reward": MyReward,

# Or at runtime:
from multi_agent_package.registry.reward_registry import register_reward
register_reward("my_reward", MyReward)
```

Config usage:
```yaml
rewards:
  functions:
    - type: my_reward
      weight: 0.5
      params:
        my_custom_param: 10
```

---

## Checklist for New Reward Functions

- [ ] Inherits from `RewardFunction`
- [ ] `compute(env)` returns `Dict[str, float]`
- [ ] No writes to env or agent state inside `compute()`
- [ ] `weight` multiplier applied in `compute()`
- [ ] Handles edge cases (e.g., no alive prey → return zeros, not errors)
- [ ] Registered in `reward_registry.py`
- [ ] Added an entry to [concepts/rewards.md](../concepts/rewards.md)
