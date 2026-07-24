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

Registered algorithms: `iql`, `cql`, `mixed` (MixedTrainer — assign IQL or CQL per team), `dqn`, `actor_critic`, `a2c`, `a3c`.

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

## ActorCritic — the On-Policy Algorithm

Unlike IQL/CQL/MixedTrainer/DQN (all value-based, act greedy-over-Q), `ActorCritic`
(`baselines/AC/actor_critic.py`) is a **policy-gradient** method: it learns a
stochastic policy directly and uses a state-value critic only to reduce gradient
variance. It is architecturally closest to DQN — one independent `ActorCriticNetwork`
plus optimizer per agent — but learns fully **on-policy**: every `_update()` call
happens immediately after the env step that produced its transition, with **no
replay buffer and no target network**.

This is the vanilla, per-step online variant — Sutton & Barto's Algorithm 13.5. The
batched (`A2C`) and asynchronous (`A3C`) variants are documented in their own
sections below.

**Same preconditions as DQN:** requires `env.observation_encoder` (raises
`ValueError` if missing), and resolves `action_dim` from
`env.action_space_plugin.n_actions` with the same fail-fast mismatch check as DQN.

**No epsilon-greedy exploration:** `select_actions()` always samples from
`Categorical(logits=...)` — the stochastic policy itself is the exploration
mechanism, so there is no `epsilon`/`epsilon_decay`/`min_epsilon` in its config.

---

## A2C — the Batched On-Policy Algorithm

`A2C` (`baselines/A2C/a2c.py`) is ActorCritic's batched sibling: same
policy-gradient idea, but accumulates an `n_steps` rollout and updates once from an
n-step bootstrapped return, instead of updating after every single env step.
Unlike ActorCritic's shared-trunk `ActorCriticNetwork`, A2C uses **separate**
`ActorNetwork`/`CriticNetwork` per agent, with independent optimizers and
independent learning rates (`actor_learning_rate`, `critic_learning_rate` — the
critic typically wants to learn faster/more stably than the actor).

**Same preconditions as ActorCritic/DQN:** `env.observation_encoder` required,
`action_dim` resolved and validated the same way.

**Config keys** (`experiment_a2c.yaml`): `gamma`, `episodes`, `hidden_layers`,
`actor_learning_rate`, `critic_learning_rate`, `n_steps`, `entropy_coef`,
`value_loss_coef`, `grad_clip`, `device`, `seed`, `curves_path`.

**Loss/optimization:** critic loss is `SmoothL1Loss` (Huber), not raw MSE — same
reasoning as DQN/ActorCritic: this environment's reward magnitudes are large
enough that plain squared error destabilizes training (empirically confirmed —
see `PROGRESS_REPORT.md` for the before/after numbers). Actor and critic each have
their own optimizer and their own `backward()`/`step()` call.

**Known limitation:** raising `entropy_coef` measurably improves capture rate on
this environment but does not prevent the policy's entropy from collapsing toward
zero mid-training — looks like a structural training-dynamics issue, not simply
"the coefficient was too small." Not yet resolved; documented for follow-up.

---

## A3C — the Asynchronous On-Policy Algorithm

`A3C` (`baselines/A3C/a3c.py`) runs A2C's same n-step rollout/update logic across
**multiple worker processes**, each stepping its own independent environment.
Workers periodically sync a local network copy from a **shared global network**
(`ActorCriticNetwork`, reused from `baselines/AC/network.py` rather than
duplicated), compute gradients locally, then apply those gradients directly to the
shared network and step a shared `SharedAdam` optimizer — lock-free ("Hogwild"),
exactly as Mnih et al. (2016) describe. There is no replay buffer and no
synchronization barrier between workers.

**Extra precondition beyond every other baseline: `config['env_fn']`.**
`BaseAlgorithm.__init__(self, env, config)` normally receives one already-built
environment; A3C needs one independent environment *per worker*, so it requires an
additional zero-argument, **picklable** callable that builds a fresh environment.
Missing or non-callable `env_fn` raises `ValueError` at construction time. A
lambda closing over a config dict will not survive pickling under the `'spawn'`
start method (the default on Windows, and used everywhere here via
`torch.multiprocessing`) — use a module-level function or a callable class
instance instead (see `scripts/run_a3c.py`'s `EnvFactory`).

**Workers always run on CPU** — not configurable. A3C's premise is parallelism
from CPU cores, not GPU batching; safely sharing one CUDA context across
processes needs CUDA IPC and isn't worth it for what this algorithm is designed
around.

**Config keys** (`experiment_a3c.yaml`): `gamma`, `episodes`, `learning_rate`,
`hidden_layers`, `num_workers`, `n_steps`, `entropy_coef`, `value_coef`,
`grad_clip`, `seed`, `curves_path` (plus the required `env_fn`, supplied by the
calling script, not the YAML).

**`SharedAdam`** (`baselines/A3C/shared_adam.py`): a `torch.optim.Adam` subclass
whose per-parameter state (`exp_avg`, `exp_avg_sq`, step count) is moved to shared
memory at construction. Without this, each worker process would keep its own
private, diverging view of the Adam moment estimates instead of a consistent
shared one.

**CSV logging across processes:** a `multiprocessing.Lock` guards concurrent
writes to `curves_path` so rows from different workers don't interleave mid-write.
Each row records a `worker` column. Episode numbers are **not** written in file
order — workers run genuinely asynchronously, so sort by the `episode` column
before any quarter/decile-style analysis rather than relying on row position.

**`A3C.save()` excludes `env_fn` from the persisted config** — it's a live
callable tied to a training session, not meaningful checkpoint data, and may not
even be picklable (e.g. a test using a lambda) even when `env_fn` itself is never
touched during training.

**Config keys** (`experiment_actor_critic.yaml`): `gamma`, `episodes`,
`learning_rate`, `hidden_layers`, `value_coef`, `entropy_coef`, `grad_clip`,
`device`, `verbose`, `log_interval`, `debug_first_episode`, `save_path`,
`curves_path`, `seed`.

**The TD error `δ` drives both networks at once:**
`δ = r + γ·(1 - terminal)·v(s') - v(s)` (true termination only, same
terminal-vs-truncated distinction as DQN's replay buffer). The critic minimizes
`δ²`; the actor's loss is `-I·δ·log π(a|s)`, where `I` starts at `1` each episode and
decays by `I *= gamma` every step (Sutton & Barto's discount-weighting term — it is
what makes this the correct on-policy gradient estimator rather than an arbitrary
TD-error weighting). `δ` is detached before entering the actor's loss, so the
policy gradient does not backpropagate through the critic's own error.

**Loss/optimization:** combined `actor_loss + value_coef * critic_loss -
entropy_coef * entropy`, one Adam optimizer per agent, gradient-clipped via
`grad_clip` — same clipping mechanism as DQN, just no `SmoothL1Loss`/Bellman-max
since there is no Q-value to take a max over.

**Behavior matches DQN, not the tabular baselines:** `train()` auto-saves to
`save_path` if configured, and supports the same `curves_path` per-episode CSV
export (columns: `episode`, `<agent>_reward`, `<agent>_loss` — no `epsilon` column,
since there is no epsilon schedule).

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
Epsilon-greedy exploration is applied **independently per agent** for the
value-based baselines (IQL, CQL, MixedTrainer, DQN). This means agents may
simultaneously explore in conflicting directions. There is no joint exploration or
coordinated strategy. In cooperative tasks, independent exploration can slow
convergence compared to approaches that coordinate exploratory actions.

`ActorCritic` explores differently: it has no epsilon schedule at all. Its
stochastic policy samples actions from `Categorical(logits=...)` directly, so
exploration is intrinsic to the policy and naturally anneals as the policy
sharpens toward confident (low-entropy) action distributions — still independent
per agent, still uncoordinated across agents, just without an explicit epsilon
knob to tune.

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
