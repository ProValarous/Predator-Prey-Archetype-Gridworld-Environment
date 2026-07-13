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

Registered algorithms: `iql`, `cql`, `mixed` (MixedTrainer — assign IQL or CQL per team), `dqn`.

---

## DQN — the Neural Algorithm

Unlike IQL/CQL/MixedTrainer (tabular), `DQN` (`baselines/DQN/dqn.py`) uses one independent PyTorch `QNetwork` (or `DuelingQNetwork`) plus a target network and a replay buffer **per agent** — architecturally, it's IQL's independent-per-agent structure with a function approximator instead of a table.

**Extra precondition beyond `BaseAlgorithm`:** DQN requires `env.observation_encoder` to already be attached — a callable `encode(obs, env) -> array-like`, flattened internally to a 1-D `float32` array. `run_from_config.build_environment()` attaches this automatically from the configured observation builder's `encode()` method; constructing `DQN` directly on an env missing this attribute raises `ValueError`.

**`action_dim` resolution differs from the tabular baselines:** DQN infers it from `env.action_space_plugin.n_actions` (falling back to `env.action_space.n` if no plugin is set) rather than taking a bare `config.get("action_dim", 5)`. If the config also sets `action_dim` explicitly and it disagrees with the inferred value, construction raises `ValueError` immediately — fail-fast instead of silently building a network with the wrong output size.

**Config keys** (`experiment_dqn.yaml`): `gamma`, `epsilon`, `epsilon_decay`, `min_epsilon`, `episodes`, `batch_size`, `buffer_size`, `min_replay_size`, `target_update_interval`, `learning_rate`, `hidden_layers`, `grad_clip`, `device`, `verbose`, `log_interval`, `debug_first_episode`, `save_path`, `curves_path`, plus two flags: `double_dqn` and `dueling` (both default `false`; the shipped `configs/experiment_dqn.yaml` doesn't set them, so it trains vanilla DQN by default — `configs/dqn_1v1/experiment_dqn.yaml` sets both `true`).

**Double DQN** (`double_dqn: true`): the bootstrap action is selected via the *online* network's argmax on `next_states`, but its Q-value is read from the *target* network — decoupling selection from evaluation to reduce the max operator's overestimation bias. Vanilla DQN (`double_dqn: false`) just takes `target_network(next_states).max()`.

**Dueling DQN** (`dueling: true`): swaps `QNetwork` for `DuelingQNetwork`, which splits into a value head `V(s)` (scalar) and an advantage head `A(s,a)` (per action), recombined as `Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))`.

**Loss/optimization:** `SmoothL1Loss` (Huber), gradient-clipped via `grad_clip`, Adam optimizer. Target networks are hard-synced (not Polyak-averaged) every `target_update_interval` optimizer steps, using a step counter shared across all agents.

**Behavioral inconsistency worth knowing:** `DQN.train()` auto-saves to `save_path` if configured. IQL/CQL/MixedTrainer's `train()` methods do **not** auto-save — saving there is a separate explicit step the caller script performs after `train()` returns.

---

## MARL Constraints and Limitations

### Non-Stationarity
Each algorithm instance sees the environment as a single-agent MDP from its perspective. In reality, other agents are also learning — their policies change every episode, making the effective transition dynamics non-stationary. This violates the stationarity assumption required for Q-learning convergence proofs.

**Implication:** Algorithms must not assume that the same observation will always lead to the same outcome. IQL, CQL, and DQN all converge empirically in small environments but have no formal convergence guarantee in multi-agent settings.

### Independent vs. Centralized Learning
**IQL** and **DQN** are fully decentralized: each agent maintains its own Q-table (or network), updated only from its own observations and rewards. No shared value function, no communication.

**CQL** is centralized: a single Q-table is shared across all agents, keyed on the joint state-action space. This enables coordinated value estimates at the cost of exponential state-space scaling with agent count. (Note: this "CQL" — Centralized Q-Learning — is unrelated to the offline-RL algorithm "Conservative Q-Learning" that shares the same acronym in the wider literature; there's no conservative/pessimistic regularization here.)

Centralized Training with Decentralized Execution (CTDE) — where a centralized critic uses global state during training but agents execute independently — is intentionally out of scope for all four baselines.

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
- [ ] Standalone CLI (`--mode train|eval`) built into the algorithm file itself, building its own env directly — this is in addition to, not instead of, a thin `run_<algo>.py` wrapper under `scripts/` that reads the matching `experiment_<algo>.yaml` via `run_from_config`'s `load_all_configs`/`build_environment`
- [ ] `train()` calls `algo.save(path)` to persist; `load(cls, env, config, path)` classmethod restores it
- [ ] Evaluation uses `algo.evaluate()` from `BaseAlgorithm` (or overrides it)
