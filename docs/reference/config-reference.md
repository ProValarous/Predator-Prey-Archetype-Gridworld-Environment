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
    speed: 1       # int — stored as agent.agent_speed; NOT used in movement loop
    stamina: 10    # int — stored as agent.stamina; NOT used in any mechanic

  preys:
    count: 3       # int — number of prey agents
    base_type: prey
    speed: 3       # int — stored but NOT used (see gotchas.md)
    stamina: 15

  naming:
    predator_prefix: predator   # str — agents named "predator_1", "predator_2", ...
    prey_prefix: prey           # str — agents named "prey_1", "prey_2", ...
```

**Notes:**
- `speed` and `stamina` are stored on agent objects but have no effect on simulation physics
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
    enabled: true      # bool — include env.base_reward() in combined reward

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

**Notes:**
- `params` is optional; `DiscreteActionSpace` accepts no parameters
- Any key not in `action_registry` raises `KeyError` at runtime
- `env.action_space_plugin.n_actions` is the authoritative source for `action_dim`; update `experiment.yaml`'s `action_dim` manually when switching action spaces

---

## experiment.yaml / experiment_iql.yaml / experiment_cql.yaml / experiment_mixed.yaml

```yaml
experiment:
  algorithm:
    name: iql          # str — must match a registered algorithm key
    params:
      # IQL / CQL shared params:
      alpha: 0.1           # float — learning rate
      gamma: 0.99          # float — discount factor
      epsilon: 1.0         # float — initial exploration rate
      epsilon_decay: 0.995 # float — multiplied by epsilon each episode
      min_epsilon: 0.05    # float — floor for epsilon
      action_dim: 5        # int — number of actions (must match env action space)
      episodes: 1000       # int — training episodes
      seed: 42             # int | null — RNG seed for action selection

      # MixedTrainer only:
      predator_algo: cql   # "iql" | "cql"
      prey_algo: iql       # "iql" | "cql"
```

**Accessing in code:**
```python
configs["experiment"]["algorithm"]["name"]   # "iql"
configs["experiment"]["algorithm"]["params"] # {...}
```
One level of `"experiment"` — `load_all_configs()` stores the parsed YAML under the key `"experiment"`, so the algorithm is at `configs["experiment"]["algorithm"]` directly.

**Available algorithm keys:** `iql`, `cql`, `mixed`
