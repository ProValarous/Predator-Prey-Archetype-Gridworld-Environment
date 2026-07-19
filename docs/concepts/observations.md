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

This is a known limitation (see [specs/environment-spec.md](../specs/environment-spec.md#known-deviations-from-spec)). For experiments where removed agents should disappear from observations, the observation builder must explicitly filter `env._captured_agents`.

---

## Observation Schema — Known Inconsistency

> ⚠️ **Current limitation:** Observation builders are not fully interchangeable. Each builder returns a dict with different top-level keys (`"agents"` vs `"visible_agents"` vs `"relative_agents"`). An algorithm written against one builder will break when switched to another without code changes. This contradicts the modularity promise. Tracked as [audit O-01](../reviews/audit-2026-06-07.md) and scheduled for fix in week 1.

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
