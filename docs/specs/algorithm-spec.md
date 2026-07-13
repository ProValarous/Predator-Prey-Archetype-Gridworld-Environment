# Spec: BaseAlgorithm

Formal contract for all learning algorithm implementations.

---

## Identity

| Property | Value |
|----------|-------|
| Abstract base | `BaseAlgorithm` |
| File | `src/baselines/base.py` |
| Registry | `baselines/registry/algorithm_registry.py` |
| Extensible by | All contributors |

---

## Required Interface

```python
class MyAlgorithm(BaseAlgorithm):
    def __init__(self, env, config: dict):
        super().__init__(env, config)
        self.env = env
        # extract hyperparameters from config
        self.alpha = config.get("alpha", 0.1)

    def select_actions(self, observations: dict) -> dict:
        # map observations to actions for all agents
        ...

    def train(self):
        # run the training loop
        ...
```

---

## `select_actions(observations)` Contract

### Input
```python
observations: Dict[str, dict]
# Keys: agent names
# Values: observation dicts (structure depends on active observation builder)
```

### Output
```python
actions: Dict[str, int]
# Keys: agent names
# Values: action indices in [0, action_dim - 1]
# Missing keys default to noop (4) in env.step()
```

### Requirements
- Must return a valid action for every agent in `observations`
- Action values must be integers in `[0, action_dim - 1]`
- May be stochastic (exploration); randomness must use a seeded RNG for reproducibility
- Must not modify `observations` in place

---

## `train()` Contract

### Requirements
- Must call only `env.reset()` and `env.step()` — no direct access to env internals
- Must call `env.reset()` at the start of each episode
- Must call `env.close()` or leave it to the caller (document which)
- Must treat the environment as a **black box** — do not read `env._obstacle_location`, `env.agents[i]._agent_location`, etc.

### No requirements on
- Logging format
- Convergence guarantee
- Memory usage

---

## Environment as Black Box

The algorithm interacts with `env` only through the step/reset API:

```python
obs, info = env.reset(seed=...)

step_out  = env.step(actions)
obs        = step_out["obs"]
rewards    = step_out["reward"]
terminated = step_out["terminated"]
truncated  = step_out["truncated"]
info       = step_out["info"]
done       = terminated or truncated

env.close()
```

Note: `env.step()` returns a **dict**, not a Gymnasium-style tuple.

It must **not** call:
- `env.reward_fn(env)` directly
- `env.observation_builder(env)` directly
- `env.base_reward()` directly
- Any private method (`env._something`)
- Direct attribute reads of agent state (`env.agents[i]._agent_location`)

**Why:** If algorithms could read agent internals, swapping observation builders would have no effect on the algorithm's behavior — defeating the purpose of the modular design.

---

## Registration

```python
# In baselines/__init__.py — import triggers self-registration:
from baselines.my_algo.my_algo import MyAlgorithm

# In my_algo.py — bottom of file (registration guard prevents double-registration
# when the module is run as __main__ via python -m):
from baselines.registry.algorithm_registry import register
if __name__ != "__main__":
    register("my_algo", MyAlgorithm)
```

Config usage:
```yaml
# experiment.yaml
experiment:
  algorithm:
    name: my_algo
    params:
      learning_rate: 0.01
      episodes: 1000
```

Registered algorithms: `iql`, `cql`, `mixed` (MixedTrainer — assign IQL or CQL per team).

---

## MARL Constraints and Limitations

### Non-Stationarity
Each algorithm instance sees the environment as a single-agent MDP from its perspective. In reality, other agents are also learning — their policies change every episode, making the effective transition dynamics non-stationary. This violates the stationarity assumption required for Q-learning convergence proofs.

**Implication:** Algorithms must not assume that the same observation will always lead to the same outcome. IQL and CQL converge empirically in small environments but have no formal convergence guarantee in multi-agent settings.

### Independent vs. Centralized Learning
**IQL** is fully decentralized: each agent maintains its own Q-table, updated only from its own observations and rewards. No shared value function, no communication.

**CQL** is centralized: a single Q-table is shared across all agents, keyed on the joint state-action space. This enables coordinated value estimates at the cost of exponential state-space scaling with agent count.

Centralized Training with Decentralized Execution (CTDE) — where a centralized critic uses global state during training but agents execute independently — is intentionally out of scope. See [ADR-004](../decisions/ADR-004-tabular-baselines.md).

### Exploration
Epsilon-greedy exploration is applied **independently per agent**. This means agents may simultaneously explore in conflicting directions. There is no joint exploration or coordinated strategy. In cooperative tasks, independent exploration can slow convergence compared to approaches that coordinate exploratory actions.

### Captured Agents
After a prey is captured, IQL/CQL continue updating its Q-table for the remainder of the episode (it still receives observations and zero-step reward). This wastes computation but does not break training — the agent is frozen and its updates do not affect the episode outcome.

---

## Checklist for New Algorithms

- [ ] Inherits from `BaseAlgorithm`
- [ ] `select_actions()` returns valid action dict
- [ ] `train()` uses only `env.reset()` / `env.step()` / `env.close()`
- [ ] No direct reads of env internals
- [ ] Hyperparameters accepted as `config: dict` in `__init__`
- [ ] Self-registers at module load via `register()` — guarded with `if __name__ != "__main__":`
- [ ] Import added to `baselines/__init__.py`
- [ ] CLI (`--mode train|eval`) built into the algorithm file itself (no separate train/eval scripts)
- [ ] `train()` calls `algo.save(path)` to persist; `load(cls, env, config, path)` classmethod restores it
- [ ] Evaluation uses `algo.evaluate()` from `BaseAlgorithm` (or overrides it)
