# System Architecture

## Layered Architecture

The system is organized into four horizontal layers. Each layer depends only on the layer below it and communicates through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4 — Scripts / Orchestration                              │
│  run_from_config.py  evaluate.py  sweep.py  render.py           │
│  Responsibility: load config, wire components, drive training   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ instantiates and wires
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 3 — Baselines (Learning Algorithms)                      │
│  baselines/IQL/  baselines/CQL/  baselines/MIXED/  baselines/DQN/│
│  Responsibility: select actions, update Q-tables/networks        │
│  Interface: BaseAlgorithm.select_actions(obs) → actions         │
│             BaseAlgorithm.train()                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ env.step(actions) / env.reset()
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 2b — Wrapper (cross-cutting mechanics)                   │
│  multi_agent_package/wrappers/speed.py — SpeedWrapper           │
│  Responsibility: honor per-agent agent_speed/stamina by         │
│                  replaying a logical step as N sub-steps.       │
│                  Applied last; proxies everything else through. │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 2 — Plugins (Observations + Rewards + Actions)           │
│  multi_agent_package/observations/   multi_agent_package/rewards/│
│  multi_agent_package/actions/                                   │
│  Responsibility: transform env state into agent percepts,        │
│                  scalar signals, and movement vectors            │
│  Interface: ObservationBuilder.build(env) → Dict[str, dict]     │
│             RewardFunction.compute(env) → Dict[str, float]      │
│             ActionSpace.to_direction(int) → np.ndarray          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ bound to env at init time
┌──────────────────────────────▼──────────────────────────────────┐
│  Layer 1 — Core Environment (IMMUTABLE)                         │
│  multi_agent_package/core/gridworld.py                          │
│  multi_agent_package/core/agent.py                              │
│  Responsibility: simulate grid physics, agent movement,         │
│                  capture detection                              │
└─────────────────────────────────────────────────────────────────┘
```

> **Quirk:** `SpeedWrapper` (Layer 2b) hardcodes its own internal `SpeedDiscreteActionSpace()` instance to decide sub-step counts — it does not read the environment's configured `action_space_plugin` (Layer 2). Both happen to treat action `4` as NOOP today, so all three shipped action spaces work correctly with it, but a future action space that used a different NOOP convention would silently break sub-stepping. See [concepts/wrappers.md](../concepts/wrappers.md).

---

## Core Package Structure

```
src/
├── multi_agent_package/          # Environment package
│   ├── core/
│   │   ├── gridworld.py          # GridWorldEnv — central simulation
│   │   └── agent.py              # Agent — identity + movement
│   ├── observations/
│   │   ├── base.py               # ObservationBuilder (abstract)
│   │   ├── default.py            # Full global view (distances)
│   │   ├── absolute.py           # World-frame coordinates
│   │   ├── relative.py           # Egocentric coordinates
│   │   ├── local_only.py         # Own position only
│   │   └── local_radius.py       # Partial: Manhattan radius filter
│   ├── rewards/
│   │   ├── base.py               # RewardFunction (abstract)
│   │   ├── base_reward.py        # Duplicates env.base_reward() — do NOT chain both
│   │   ├── predator_distance.py  # -weight * manhattan_dist_to_nearest_prey
│   │   └── survival_reward.py    # +weight per step for prey
│   ├── actions/
│   │   ├── base.py               # ActionSpace (abstract)
│   │   ├── discrete_actions.py   # DiscreteActionSpace — 5 actions (R/U/L/D/Noop)
│   │   ├── cross_actions.py      # CrossActionSpace — 4 diagonals + Noop
│   │   └── speed_discrete.py     # SpeedDiscreteActionSpace — discrete_5 + to_moves()
│   ├── wrappers/
│   │   ├── __init__.py           # exports SpeedWrapper
│   │   └── speed.py              # SpeedWrapper — per-agent speed/stamina sub-stepping
│   ├── registry/
│   │   ├── observation_registry.py
│   │   ├── reward_registry.py
│   │   └── action_registry.py
│   └── scripts/
│       ├── run_from_config.py    # Generic entrypoint (algorithm chosen via YAML)
│       ├── run_iql.py  run_cql.py  run_mixed.py  run_dqn.py   # per-algorithm train/eval CLIs
│       ├── evaluate.py           # Metrics collection (episode length + per-agent return)
│       ├── render.py             # Single-episode visualization (delegates to run_from_config)
│       └── sweep.py              # Hardcoded sweep of observations.yaml's "radius" param
│
├── baselines/
│   ├── base.py                   # BaseAlgorithm (abstract) + evaluate()
│   ├── __init__.py               # Auto-registers IQL, CQL, MixedTrainer, DQN
│   ├── IQL/
│   │   └── iql.py                # IQL class + standalone CLI (python -m baselines.IQL.iql)
│   ├── CQL/
│   │   └── cql.py                # CQL class + standalone CLI (python -m baselines.CQL.cql)
│   ├── MIXED/
│   │   └── mix_train.py          # MixedTrainer class + CLI (per-team IQL/CQL)
│   ├── DQN/
│   │   ├── dqn.py                # DQN class + standalone CLI (Double DQN, Dueling DQN)
│   │   ├── q_network.py          # QNetwork / DuelingQNetwork (PyTorch)
│   │   └── replay_buffer.py      # Fixed-capacity numpy ring buffer
│   └── registry/
│       └── algorithm_registry.py
│
configs/
├── env.yaml  agents.yaml  observations.yaml  rewards.yaml  actions.yaml
├── experiment.yaml          # default (IQL, 3v3)
├── experiment_iql.yaml  experiment_cql.yaml  experiment_mixed.yaml  experiment_dqn.yaml
└── dqn_1v1/  dqn_speed1/  dqn_speed2/  dqn_speed3/   # ready-made DQN experiment sets

tests/
├── conftest.py                    # shared fixtures
├── test_core_agent.py             # Agent identity, directions, obs/info
├── test_core_gridworld.py         # GridWorldEnv reset, step, capture, truncation
├── test_observations.py           # all 5 observation builders
├── test_rewards.py                # all 3 reward functions
├── test_actions.py                # all 3 action spaces
├── test_registry.py               # observation, reward, action, and algorithm registries
├── test_baselines_iql.py  test_baselines_cql.py  test_baselines_mixed.py  test_baselines_dqn.py
├── test_architecture_contract.py  # plugin ABC contracts, no-mutation checks, config-pipeline sweep
└── test_integration.py            # config loading, build_environment, end-to-end training
```

---

## Component Interaction at Runtime

### Initialization sequence

```
1. run_from_config.main(config_dir)
   │
   ├── load_all_configs()          reads env/agents/observations/rewards/actions/experiment YAML
   │
   ├── build_agents(agent_cfg)     creates Agent instances, sets agent_speed/stamina as attributes
   │
   ├── build_environment(configs)
   │   ├── GridWorldEnv(agents, **env_params)
   │   ├── get_observation_builder(type, **params)  → builder
   │   │   env.observation_builder = builder.build
   │   │   env.observation_encoder = builder.encode   (required by DQN)
   │   ├── [get_reward_function(name, weight) for each shaping entry]
   │   │   combined = closure summing all reward outputs (base_reward() is NOT re-added here —
   │   │   gridworld.step() already computes it internally)
   │   │   env.reward_fn = combined
   │   ├── get_action_space(type, **params)  → space
   │   │   env.action_space_plugin = space
   │   └── env = SpeedWrapper(env)   ← wraps LAST, so the plugins above are already attached
   │       (SpeedWrapper proxies unknown attributes through via __getattr__)
   │
   └── AlgorithmClass(env, **hyper_params).train()
```

### Per-step sequence

```
algorithm.select_actions(obs)  →  actions: Dict[str, int]
          │
          ▼
SpeedWrapper.step(actions)   ← only if some agent's agent_speed > 1
   ├── compute each agent's sub-step budget via its own SpeedDiscreteActionSpace().to_moves(...)
   ├── replay up to max_speed sub-steps, sending NOOP once an agent's budget is used
   ├── sum rewards across sub-steps; break early on terminated/truncated
   ├── deduct 1 stamina per sub-step taken
   └── returns the FINAL sub-step's obs/terminated/truncated, SUMMED rewards
          │
          ▼  (each sub-step calls the underlying env:)
GridWorldEnv.step(sub_actions)
   ├── for each agent: action_space_plugin.to_direction(a) ← plugged-in action space
   ├── validate moves (bounds + obstacles)
   ├── apply positions simultaneously
   ├── detect captures (predator+prey same cell)
   ├── rewards = self.base_reward()          ← always runs, hardcoded
   ├── env.reward_fn(env)                    ← plugged-in shaping closure, added on top
   ├── check termination/truncation
   ├── env.observation_builder(env)          ← plugged-in builder
   └── return {"obs","reward","terminated","truncated","info"}  (a dict, not a tuple)
          │
          ▼
algorithm  (updates internal Q-tables / replay buffer + networks via train loop)
```

---

## Extension Points

The architecture has four explicit plugin extension points, plus wrappers as a less common fifth:

| Extension Point | Where | How |
|----------------|-------|-----|
| Custom observation | `observations/` | Subclass `ObservationBuilder`, implement `build(env)`, register in `observation_registry.py` |
| Custom reward | `rewards/` | Subclass `RewardFunction`, implement `compute(env)`, register in `reward_registry.py` |
| Custom action space | `actions/` | Subclass `ActionSpace`, implement `to_direction()` + properties, register in `action_registry.py` (note: `register_action_space()` does **not** validate the class is a subclass, unlike the other two registries) |
| Custom algorithm | `baselines/` | Subclass `BaseAlgorithm`, implement `select_actions()` + `train()`, register in `algorithm_registry.py` |
| Custom wrapper | `wrappers/` | Follow `SpeedWrapper`'s pattern: wrap an env, proxy unmodified attributes via `__getattr__`, override only `step()`/`reset()`. No registry exists for wrappers — they're applied explicitly in `build_environment()`. |

---

## Dependency Graph

```
run_from_config.py
    ├── gridworld.py  ←  agent.py
    ├── observation_registry.py  ←  [Default, LocalOnly, LocalRadius, Absolute, Relative]
    ├── reward_registry.py       ←  [BaseReward, PredatorDistanceReward, SurvivalReward]
    ├── action_registry.py       ←  [DiscreteActionSpace, CrossActionSpace, SpeedDiscreteActionSpace]
    ├── wrappers/speed.py        ←  SpeedWrapper (wraps the fully-built env last)
    └── algorithm_registry.py   ←  [IQL, CQL, MixedTrainer, DQN]

gridworld.py
    └── agent.py  (holds agent state; env orchestrates it)

wrappers/speed.py
    └── actions/speed_discrete.py  (SpeedDiscreteActionSpace, hardcoded — not env.action_space_plugin)

IQL / CQL / MixedTrainer / DQN
    └── gridworld.py (or SpeedWrapper-wrapped gridworld.py)  — black-box: only reset() and step()
```

No circular dependencies. The core never imports from plugins or baselines.
