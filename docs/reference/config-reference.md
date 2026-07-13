# Config Reference

Every YAML key accepted by the system, with types, defaults, and notes.

---

## env.yaml

```yaml
env:
  name: gridworld            # str — display name only, not used in code
  size: 10                   # int — grid is size×size cells
  obstacle_percentage: 20.0  # float — % of cells blocked (0–100)
  seed: 42                   # int | null — master RNG seed; null = random
  render_mode: null          # "human" | "rgb_array" | null
  window_size: 600           # int — pygame window size in pixels (human mode only)

  dynamics:
    allow_cell_sharing: true         # bool — multiple agents can share a cell
    block_agents_by_obstacles: true  # bool — agents cannot move into obstacle cells

  termination:
    capture_threshold: 1     # int — episode ends after this many total captures
    max_steps: 500           # int | null — episode ends after this many steps; null = no limit
```

**Notes:**
- `obstacle_percentage: 0` creates a clear grid
- `size` must be large enough to place all agents without collision: `size^2 > n_agents + n_obstacles`
- `capture_threshold: 2` with only 1 prey → episode can never terminate via capture (truncated only)
- `render_mode: "rgb_array"` is accepted but returns `None` (not implemented)

---

## agents.yaml

```yaml
agents:
  predators:
    count: 3       # int — number of predator agents
    base_type: predator
    speed: 1       # int — stored as agent.agent_speed
    stamina: 10    # int — stored as agent.stamina

  preys:
    count: 3       # int — number of prey agents
    base_type: prey
    speed: 3       # int
    stamina: 15

  naming:
    predator_prefix: predator   # str — agents named "predator_1", "predator_2", ...
    prey_prefix: prey           # str — agents named "prey_1", "prey_2", ...
```

**Notes:**
- `GridWorldEnv.step()` itself ignores `speed`/`stamina` entirely — it always moves every agent exactly one cell. They matter because `run_from_config.build_environment()` always wraps the env in `SpeedWrapper`, which reads both to decide each agent's sub-step budget per logical step and to deplete stamina over the episode. See [concepts/wrappers.md](../concepts/wrappers.md).
- Agent names are `f"{prefix}_{i+1}"` for i in range(count)
- Total agents = predators.count + preys.count; must fit in the grid

---

## observations.yaml

```yaml
observations:
  type: local_radius       # str — registry key; see table below
  params:                  # dict — passed as **kwargs to builder's __init__
    radius: 3
    include_agents: true
    include_obstacles: true
  per_agent:               # IGNORED — accepted by YAML but not wired in code
    predator:
      radius: 4
    prey:
      radius: 2
```

**Available observation types:**

| Key | Builder | Notable params |
|-----|---------|----------------|
| `default` | `DefaultObservation` | none |
| `local_only` | `LocalOnlyObservation` | none |
| `local_radius` | `LocalRadiusObservation` | `radius: int`, `include_agents: bool`, `include_obstacles: bool` |
| `absolute` | `AbsoluteObservation` | none |
| `relative` | `RelativeObservation` | `include_agents: bool`, `include_obstacles: bool`, `include_walls: bool`, `distance_type: "manhattan"\|"euclidean"` |

**Notes:**
- `per_agent` block is silently ignored — all agents receive the same params
- Any key not in the registry raises `KeyError` at runtime

---

## rewards.yaml

```yaml
rewards:
  base:
    enabled: true      # bool — IGNORED (see note below); base_reward() always runs regardless

  shaping:             # list — zero or more shaping functions
    - name: predator_distance   # str — registry key
      weight: 0.5               # float — multiplier applied to this function's output
      applies_to: predator      # str — IGNORED; all functions apply to all agents

    - name: survival
      weight: 1.0
      applies_to: prey          # IGNORED

  normalization:
    enabled: false     # bool — IGNORED; not implemented
```

**Available reward functions:**

| Key | Class | Gives reward to |
|-----|-------|----------------|
| `base` | `BaseReward` | All agents — wraps `env.base_reward()` |
| `predator_distance` | `PredatorDistanceReward` | Predators — `−weight × dist_to_nearest_prey` |
| `survival` | `SurvivalReward` | Prey — `+weight` per step |

**Base reward hardcoded values** (not configurable via YAML):

| Event | Reward |
|-------|--------|
| Predator captures prey | +100 (predator) |
| Prey captured | −100 (prey) |
| Each predator per step | −5 (step cost) |
| Agent on obstacle cell | −200 |

**Notes:**
- `applies_to` is parsed but never read — filter by type inside `compute()` if needed
- `normalization.enabled` field is accepted but not implemented
- `base.enabled` is also inert post [PR #26](https://github.com/ProValarous/Predator-Prey-Archetype-Gridworld-Environment/pull/26): `GridWorldEnv.step()` calls `self.base_reward()` unconditionally every step, outside the plugin system entirely. `run_from_config.py` reads this key only to assert it exists — the boolean value is discarded. There's a registered `"base"` reward-function class that wraps `env.base_reward()`, but it's deliberately **not** added to the active `shaping`/reward-function chain, because doing so would double-count every base signal (that was a real bug, now fixed).
- Shaping rewards **add** to base reward; they don't replace it

---

## actions.yaml

```yaml
actions:
  type: discrete_5   # str — registry key; see table below
  params: {}         # dict — passed as **kwargs to ActionSpace.__init__
```

**Available action space types:**

| Key | Class | Actions |
|-----|-------|---------|
| `discrete_5` | `DiscreteActionSpace` | RIGHT `[+1,0]`, UP `[0,+1]`, LEFT `[-1,0]`, DOWN `[0,-1]`, NOOP `[0,0]` |
| `cross` | `CrossActionSpace` | NE `[+1,+1]`, NW `[-1,+1]`, SW `[-1,-1]`, SE `[+1,-1]`, NOOP `[0,0]` — diagonal-only, no cardinal moves |
| `speed_discrete_5` | `SpeedDiscreteActionSpace` | Same 5 actions as `discrete_5` (subclasses it); adds a `to_moves(action, speed, stamina)` method used by `SpeedWrapper` for sub-step counting |

**Notes:**
- `params` is optional; none of the three shipped action spaces accept constructor params
- Any key not in `action_registry` raises `KeyError` at runtime
- `env.action_space_plugin.n_actions` is the authoritative source for `action_dim`; update `experiment.yaml`'s `action_dim` manually when switching action spaces for IQL/CQL/MixedTrainer (DQN infers and validates it automatically — see `reference/api-reference.md`)
- `SpeedWrapper` does not read this config's chosen action space at all for its own sub-stepping — it always uses its own internal `SpeedDiscreteActionSpace()` instance regardless of what's configured here (see [concepts/wrappers.md](../concepts/wrappers.md))
- `register_action_space()` (for custom action spaces) does not validate `issubclass`, unlike the observation/reward registries

---

## experiment.yaml / experiment_iql.yaml / experiment_cql.yaml / experiment_mixed.yaml / experiment_dqn.yaml

```yaml
experiment:
  algorithm:
    name: iql          # str — must match a registered algorithm key: iql | cql | mixed | dqn
    params:
      # IQL / CQL / MixedTrainer shared params:
      alpha: 0.1           # float — learning rate
      gamma: 0.99          # float — discount factor
      epsilon: 1.0         # float — initial exploration rate
      epsilon_decay: 0.995 # float — multiplied by epsilon each episode
      min_epsilon: 0.05    # float — floor for epsilon
      action_dim: 5        # int — number of actions; NOT validated against the env's action space for these three
      episodes: 1000       # int — training episodes
      seed: 42             # int | null — RNG seed for action selection

      # MixedTrainer only:
      predator_algo: cql   # "iql" | "cql"
      prey_algo: iql       # "iql" | "cql"
```

```yaml
experiment:
  algorithm:
    name: dqn
    params:
      # gamma, epsilon, epsilon_decay, min_epsilon, episodes, seed: same meaning as above
      batch_size: 32               # int — samples per gradient step
      buffer_size: 10000           # int — replay buffer capacity
      min_replay_size: 32          # int — defaults to batch_size; buffer must reach this before training starts
      target_update_interval: 100  # int — optimizer steps between hard target-network syncs
      learning_rate: 0.001         # float — Adam learning rate
      hidden_layers: [64, 64]      # list[int] — QNetwork/DuelingQNetwork hidden layer sizes
      grad_clip: 5.0               # float — gradient norm clip
      device: "cpu"                # str — torch.device string
      double_dqn: false            # bool — decouple action selection (online net) from evaluation (target net)
      dueling: false               # bool — split network into value + advantage streams
      curves_path: null            # str | null — CSV path for per-episode reward/loss/epsilon logging
      # action_dim: intentionally omitted here — DQN infers it from env.action_space_plugin.n_actions
      #             and raises ValueError if you set it explicitly and it disagrees
```

**Accessing in code:**
```python
configs["experiment"]["experiment"]["algorithm"]["name"]     # "iql"
configs["experiment"]["experiment"]["algorithm"]["params"]   # {...}
```
`load_all_configs()` stores the parsed YAML under the key `"experiment"`, and the YAML file's own content is itself `experiment:\n  algorithm: ...` — so the outer `"experiment"` (from `load_all_configs`) wraps the inner `"experiment"` (the YAML's own top-level key), giving the double-nested path above. Verified directly against `run_from_config.py`'s `main()`.

**Available algorithm keys:** `iql`, `cql`, `mixed`, `dqn`

**Ready-made DQN experiment sets** (full `env`/`agents`/`observations`/`rewards`/`actions`/`experiment_dqn` YAML each) live as subdirectories rather than root-level files:

| Directory | Setup |
|-----------|-------|
| `configs/dqn_1v1/` | 1 predator (speed 2) vs 1 prey (speed 1), 10×10 grid, 20% obstacles, `double_dqn`+`dueling` both enabled |
| `configs/dqn_speed1/` | 1v1, both agents speed 1 (baseline, no speed advantage) |
| `configs/dqn_speed2/` | 1v1, predator speed 2 |
| `configs/dqn_speed3/` | 1v1, predator speed 3 |

Run any of them with `PYTHONPATH=src python -m multi_agent_package.scripts.run_dqn --config-dir configs/dqn_1v1`.
