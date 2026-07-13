# Config Recipes

Common experiment configurations. Copy the relevant sections into your YAML files.

---

## Fully blind agents (local position only)

Agents see only their own `[x, y]`. No information about other agents or obstacles. Maximally challenging for learning.

```yaml
# observations.yaml
observations:
  type: local_only
  params: {}
```

---

## Full global view (default)

Each agent sees Euclidean distances to all other agents and all obstacles.

```yaml
observations:
  type: default
  params: {}
```

---

## Partial observability — radius-based

Agents see other agents and obstacles within Manhattan radius `r`. Good balance between tractability and realism.

```yaml
observations:
  type: local_radius
  params:
    radius: 3
    include_agents: true
    include_obstacles: true
```

Larger radius = more information = slower Q-table growth. Keep `radius ≤ 4` for tabular methods on a 10×10 grid.

---

## Egocentric (relative) view

All positions expressed relative to the observing agent. Agent always sees itself at `[0, 0]`.

```yaml
observations:
  type: relative
  params:
    include_agents: true
    include_obstacles: true
    include_walls: false
    distance_type: manhattan   # or "euclidean"
```

---

## Sparse reward only (capture/death signals)

No step cost, no shaping. Pure capture-based signal. Hardest for learning, most informative for ablation.

```yaml
# rewards.yaml
rewards:
  base:
    enabled: true   # +100 predator capture, -100 prey captured
  shaping: []
```

Note: base reward always includes a −5 step cost for predators. To remove it, you'd need a custom reward function wrapping `base_reward()` with modified values — currently the step cost is hardcoded.

---

## Dense shaping — predator distance

Predators get negative reward proportional to distance to nearest prey. Encourages approach.

```yaml
rewards:
  base:
    enabled: true
  shaping:
    - name: predator_distance
      weight: 0.5
```

Tune `weight` up if predators are not approaching; tune down if they learn approach but not capture.

---

## Prey survival reward

Prey get +`weight` per step for staying alive. Adds an explicit survival incentive.

```yaml
rewards:
  base:
    enabled: true
  shaping:
    - name: survival
      weight: 1.0
```

---

## Dense for both sides

```yaml
rewards:
  base:
    enabled: true
  shaping:
    - name: predator_distance
      weight: 0.5
    - name: survival
      weight: 1.0
```

---

## Long episodes (exploration-heavy)

Good for large grids or sparse reward settings.

```yaml
# env.yaml
termination:
  capture_threshold: 1
  max_steps: 1000
```

---

## Short episodes (fast iteration)

Good for debugging and hyperparameter search.

```yaml
termination:
  capture_threshold: 1
  max_steps: 100
```

---

## Multi-capture termination

Episode continues until `N` captures. Useful for studying predator coordination.

```yaml
termination:
  capture_threshold: 3   # episode ends after 3 prey captured
  max_steps: 500
```

---

## Obstacle-free grid

Remove all obstacles. Cleaner dynamics for baseline experiments.

```yaml
env:
  obstacle_percentage: 0.0
```

---

## High obstacle density

Dense obstacle grid. Tests path planning and obstacle-avoidance incidentally learned through capture rewards.

```yaml
env:
  obstacle_percentage: 40.0
```

> With `block_agents_by_obstacles: true` (default), 40% obstacles on a 10×10 grid leaves 60 traversable cells for 6 agents — very tight.

---

## Asymmetric team sizes

```yaml
# agents.yaml
agents:
  predators:
    count: 1
  preys:
    count: 3
```

One predator versus three prey. Predator must learn to corner. Prey can disperse.

---

## CQL — keep state space small

CQL scales as `action_dim^n_agents` per joint state. On a large grid with many agents, the table becomes huge. Recommended settings for tabular CQL:

```yaml
# env.yaml
env:
  size: 6            # not 10
agents:
  predators:
    count: 1         # not 3
  preys:
    count: 1
```

With `size=6, 1 pred, 1 prey, 5 actions`: joint action space = 5^2 = 25 per state. Manageable.

---

## Speed asymmetry (predator faster than prey)

Requires `speed_discrete_5` or any action space paired with `SpeedWrapper`'s sub-stepping (speed comes from `agents.yaml`, not from the action space itself):

```yaml
# agents.yaml
agents:
  predators:
    speed: 2       # via SpeedWrapper: up to 2 sub-steps per logical turn
  preys:
    speed: 1
```

See `configs/dqn_speed1`, `dqn_speed2`, `dqn_speed3` for a ready-made 1/2/3 speed sweep, and [concepts/wrappers.md](../concepts/wrappers.md) for the mechanics.

---

## Diagonal-only movement

```yaml
# actions.yaml
actions:
  type: cross    # NE/NW/SW/SE + noop — no cardinal moves at all
  params: {}
```

Remember to update `action_dim` in your experiment YAML (still 5, since `cross` is also a 5-action space) if it's set explicitly.

---

## Double DQN + Dueling DQN

```yaml
# experiment_dqn.yaml
experiment:
  algorithm:
    name: dqn
    params:
      double_dqn: true   # decouples action selection (online net) from evaluation (target net)
      dueling: true      # splits the network into value + advantage streams
```

See `configs/dqn_1v1/experiment_dqn.yaml` for a working example with both enabled.
