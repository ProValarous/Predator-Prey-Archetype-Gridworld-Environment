# Concept: Observations

## What an Observation Is

An observation is a **per-agent dict** returned after each step that tells each agent what it currently perceives about the world. The observation system transforms raw environment state into a structured percept appropriate for the chosen observability model.

Observations are the only information a learning algorithm receives about the world. The algorithm cannot see `env.agents` or `env._obstacle_location` directly — it only sees what the observation builder provides.

---

## Observability Spectrum

```
Minimal                                                      Full
  │                                                            │
LocalOnly ──── LocalRadius ──── Default ──── Absolute ──── Relative
  │                │                │              │              │
Only own     Nearby entities   Distances     Absolute       Egocentric
position     within radius     to all        coordinates    coordinates
```

| Builder | Key | Sees | Frame |
|---------|-----|------|-------|
| `LocalOnlyObservation` | `local_only` | Own position only | Absolute |
| `LocalRadiusObservation` | `local_radius` | Entities within Manhattan radius | Hybrid |
| `DefaultObservation` | `default` | Euclidean distances to all entities | Mixed |
| `AbsoluteObservation` | `absolute` | All positions in world coordinates | World |
| `RelativeObservation` | `relative` | All positions relative to self | Egocentric |

---

## Two Observation Frames

**World Frame (Absolute)**
Positions are in grid coordinates. `[0,0]` is always top-left. An agent at `[3, 2]` sees other agents at their literal grid positions.

**Egocentric Frame (Relative)**
The observing agent is always at `[0, 0]`. All other positions are expressed as offsets `(dx, dy)`. An agent at `[3, 2]` sees an entity at `[5, 4]` as `[+2, +2]`.

Egocentric observations are **translationally invariant**: a policy trained in one area of the grid should generalize to other areas. World-frame observations are not.

---

## Partial vs. Full Observability

**Full observability**: the agent sees all other entities regardless of distance. Used by `Default`, `Absolute`, and `Relative` builders.

**Partial observability** (`LocalRadius`): the agent only sees entities within a Manhattan radius `r`. Entities further than `r` are invisible. This creates a realistic sensor model and is the default config.

**Minimal observability** (`LocalOnly`): the agent sees nothing except its own position. Used as a blind-agent control condition.

---

## Captured Agents in Observations

Captured prey are **not removed** from observations after capture. Their position is frozen but they continue to appear in every observation dict with their last known location. This means:

- Predators still see captured prey in their observation (they appear as stationary entities)
- Prey still see captured teammates in their observation
- Q-tables may grow entries for states that include frozen prey positions

This is a known limitation of the current implementation. For experiments where removed agents should disappear from observations, the observation builder must explicitly filter `env._captured_agents`.

---

## Observation Schema — Known Inconsistency

> ⚠️ **Current limitation:** Observation builders are not fully interchangeable. Each builder returns a dict with different top-level keys for other entities — `dist_agents`/`dist_obstacles` (`default`), `visible_agents`/`visible_obstacles` (`local_radius`), `agents`/`obstacles` (`absolute` and `relative`) — see the [Output Schema per Builder](#output-schema-per-builder) section below for exact shapes. An algorithm or observation-consuming script written against one builder's schema will break when switched to another without code changes. This contradicts the modularity promise, and there's no adapter layer that normalizes across builders today.

---

## The Builder Interface

Every observation builder is a class that:
1. Accepts config params at construction (radius, distance_type, etc.)
2. Exposes a `build(env)` method that returns `Dict[str, dict]`
3. Is **read-only** — it may not modify env or agent state

The `build` method is bound to the environment at init time:
```python
env.observation_builder = builder_instance.build
```

During `step()`, the env calls `self.observation_builder(self)` — passing itself as the only argument.

For the full contract, see [specs/observation-builder-spec.md](../specs/observation-builder-spec.md).

---

## Observations and Learning

Observations are converted to Q-table keys by the learning algorithm's `_encode_state()`. This means:

- Complex observations (many visible entities) → large tuple keys → sparse Q-table coverage
- Simple observations (LocalOnly, radius=1) → small tuple keys → faster convergence but less information
- The observation choice directly affects how tractable tabular learning is

For a 10×10 grid with LocalOnly observations, the state space per agent is 100 positions. With full Absolute observations (all 6 agents + ~12 obstacles), the state tuple is much longer and the effective state space is enormous.

---

## Choosing an Observation Builder

| Goal | Recommended Builder |
|------|-------------------|
| Control condition (blind agent) | `local_only` |
| Realistic sensor model | `local_radius` (radius=2–4) |
| Full-information baseline | `default` or `absolute` |
| Translation-invariant policy | `relative` |
| Ablate information radius | `local_radius` with varying `radius` |

---

## Output Schema per Builder

Each builder's exact return shape (per agent). These differ across builders — see the "Known Inconsistency" note above.

### `DefaultObservation` (`default`)

```python
{
    "agent_name": {
        "local": np.ndarray,        # own position [x, y]
        "global": {
            "dist_agents": {"other_agent_name": float, ...},      # Euclidean
            "dist_obstacles": {"obstacle_0": float, ...},          # Euclidean
        }
    }
}
```
Delegates to `GridWorldEnv._default_observations()`. No constructor params.

### `LocalOnlyObservation` (`local_only`)

```python
{
    "agent_name": {
        "local": np.ndarray,   # own position [x, y]
        "global": None,
    }
}
```
No constructor params. Reads only `agent._agent_location`.

### `LocalRadiusObservation` (`local_radius`)

Params: `radius: int = 3`, `include_agents: bool = True`, `include_obstacles: bool = True`.

```python
{
    "agent_name": {
        "local": np.ndarray,
        "visible_agents": {
            "other_agent_name": {"rel_pos": tuple, "dist": int, "type": str}
        },
        "visible_obstacles": {
            "obstacle_i": {"rel_pos": tuple, "dist": int}
        },
        "radius": int,
    }
}
```
Distance is Manhattan; only entities with `dist <= radius` appear.

### `AbsoluteObservation` (`absolute`)

Params: `include_agents: bool = True`, `include_obstacles: bool = True`, `distance_type: "euclidean"|"manhattan" = "euclidean"`.

```python
{
    "agent_name": {
        "local": {"pos": np.array([x, y]), "type": str, "team": str, "speed": int},
        "agents": {
            "other_agent_name": {"pos": np.array([x, y]), "dist": float, "type": str, "team": str}
        },
        "obstacles": {
            "obstacle_0": {"pos": np.array([x, y]), "dist": float}
        }
    }
}
```
All positions are world-frame (absolute grid coordinates).

### `RelativeObservation` (`relative`)

Params: `include_agents: bool = True`, `include_obstacles: bool = True`, `include_walls: bool = False`, `distance_type: "manhattan"|"euclidean" = "manhattan"`.

```python
{
    "agent_name": {
        "local": {"pos": np.array([0, 0]), "type": str, "team": str, "speed": int},  # ALWAYS origin
        "agents": {
            "other_agent_name": {"rel_pos": np.array([dx, dy]), "dist": int, "type": str, "team": str}
        },
        "obstacles": {
            "obstacle_0": {"rel_pos": np.array([dx, dy]), "dist": int}
        },
        "walls": {"left": int, "right": int, "up": int, "down": int}   # if include_walls=True
    }
}
```
The observing agent is always reported at `[0, 0]`; every other position is an offset relative to it.
