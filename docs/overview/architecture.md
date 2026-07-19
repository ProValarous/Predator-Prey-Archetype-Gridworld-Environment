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
│  baselines/IQL/   baselines/CQL/   baselines/MIXED/             │
│  Responsibility: select actions, update Q-tables                │
│  Interface: BaseAlgorithm.select_actions(obs) → actions         │
│             BaseAlgorithm.train()                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ env.step(actions) / env.reset()
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
│   │   ├── base_reward.py        # Wraps hardcoded env rewards
│   │   ├── predator_distance.py  # -dist_to_nearest_prey shaping
│   │   └── survival_reward.py    # +weight per step for prey
│   ├── actions/
│   │   ├── base.py               # ActionSpace (abstract)
│   │   └── discrete_actions.py   # DiscreteActionSpace — 5 actions (R/U/L/D/Noop)
│   ├── registry/
│   │   ├── observation_registry.py
│   │   ├── reward_registry.py
│   │   └── action_registry.py
│   └── scripts/
│       ├── run_from_config.py    # Main entrypoint
│       ├── evaluate.py           # Metrics collection (per-agent returns + ep lengths)
│       ├── render.py             # Single-episode visualization
│       └── sweep.py              # Parameter sweep harness
│
├── baselines/
│   ├── base.py                   # BaseAlgorithm (abstract) + evaluate()
│   ├── __init__.py               # Auto-registers IQL, CQL, MixedTrainer
│   ├── IQL/
│   │   └── iql.py                # IQL class + CLI  (python -m baselines.IQL.iql)
│   ├── CQL/
│   │   └── cql.py                # CQL class + CLI  (python -m baselines.CQL.cql)
│   ├── MIXED/
│   │   └── mix_train.py          # MixedTrainer class + CLI (per-team IQL/CQL)
│   └── registry/
│       └── algorithm_registry.py
│
configs/
├── env.yaml
├── agents.yaml
├── observations.yaml
├── rewards.yaml
├── actions.yaml
├── experiment.yaml          # default (IQL)
├── experiment_iql.yaml
├── experiment_cql.yaml
└── experiment_mixed.yaml

tests/
├── conftest.py              # shared fixtures
├── test_core_agent.py       # Agent identity, directions, obs/info
├── test_core_gridworld.py   # GridWorldEnv reset, step, capture, truncation
├── test_observations.py     # all 5 observation builders
├── test_rewards.py          # all 3 reward functions
├── test_registry.py         # observation, reward, and algorithm registries
├── test_baselines_iql.py    # IQL: Q-tables, encoding, select_actions, save/load
├── test_baselines_cql.py    # CQL: joint state/action, marginalisation, save/load
├── test_baselines_mixed.py  # MixedTrainer: team partitioning, IQL/CQL modes
└── test_integration.py      # config loading, build_environment, end-to-end training
```

---

## Component Interaction at Runtime

### Initialization sequence

```
1. run_from_config.main(config_dir)
   │
   ├── load_all_configs()          reads 5 YAML files
   │
   ├── build_agents(agent_cfg)     creates Agent instances
   │
   ├── build_environment(configs)
   │   ├── GridWorldEnv(agents, **env_params)
   │   ├── get_observation_builder(type, **params)  → builder
   │   │   env.observation_builder = builder.build
   │   ├── [get_reward_function(name, weight) for each]
   │   │   combined = closure summing all reward outputs
   │   │   env.reward_fn = combined
   │   ├── get_action_space(type, **params)  → space
   │   │   env.action_space_plugin = space
   │   └── returns wired env
   │
   └── AlgorithmClass(env, **hyper_params).train()
```

### Per-step sequence

```
algorithm.select_actions(obs)  →  actions: Dict[str, int]
          │
          ▼
env.step(actions)
   ├── for each agent: action_space_plugin.to_direction(a) ← plugged-in action space
   ├── validate moves (bounds + obstacles)
   ├── apply positions simultaneously
   ├── detect captures (predator+prey same cell)
   ├── env.reward_fn(env)           ← plugged-in reward closure
   ├── check termination
   ├── env.observation_builder(env) ← plugged-in builder
   └── return (obs, rewards, terminated, truncated, info)
          │
          ▼
algorithm  (updates internal Q-tables via train loop)
```

---

## Extension Points

The architecture has four explicit extension points:

| Extension Point | Where | How |
|----------------|-------|-----|
| Custom observation | `observations/` | Subclass `ObservationBuilder`, implement `build(env)`, register in `observation_registry.py` |
| Custom reward | `rewards/` | Subclass `RewardFunction`, implement `compute(env)`, register in `reward_registry.py` |
| Custom action space | `actions/` | Subclass `ActionSpace`, implement `to_direction()` + properties, register in `action_registry.py` |
| Custom algorithm | `baselines/` | Subclass `BaseAlgorithm`, implement `select_actions()` + `train()`, register in `algorithm_registry.py` |

---

## Dependency Graph

```
run_from_config.py
    ├── gridworld.py  ←  agent.py
    ├── observation_registry.py  ←  [all observation builders]
    ├── reward_registry.py       ←  [all reward functions]
    ├── action_registry.py       ←  [DiscreteActionSpace, ...]
    └── algorithm_registry.py   ←  [IQL, CQL, MixedTrainer]

gridworld.py
    └── agent.py  (holds agent state; env orchestrates it)

IQL / CQL / MixedTrainer
    └── gridworld.py  (black-box: only reset() and step())
```

No circular dependencies. The core never imports from plugins or baselines.
