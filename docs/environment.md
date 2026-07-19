# GridWorld Environment

## Overview

The `GridWorldEnv` class implements a discrete, bounded 2D grid where predator
and prey agents interact. It serves as the "game board" that manages all agents,
obstacles, movement, captures, and rewards.

This page explains the concepts. For method details, see the
[API Reference](api/gridworld.md).

---

## Coordinate System

The grid uses a standard 2D coordinate system:



### Key Points

| Concept | Description |
|---------|-------------|
| **Origin** | `[0, 0]` is the **top-left** corner |
| **X-axis** | Increases to the **right** |
| **Y-axis** | Increases **downward** |
| **Position** | Stored as `[x, y]` numpy array |
| **Grid size** | Configurable via `size` parameter (default: 5×5) |

### Position Examples

```python
# Agent positions as [x, y]
top_left     = [0, 0]
top_right    = [4, 0]  # For a 5×5 grid
bottom_left  = [0, 4]
bottom_right = [4, 4]
center       = [2, 2]
```

## Actions and Movements 

Each agent can perform one of 5 discrete actions per timestep:

| Action | Index | Direction Vector | Description |
|--------|-------|------------------|-------------|
| **Right** | 0 | `[+1, 0]` | Move one cell right |
| **Up** | 1 | `[0, -1]` | Move one cell up (toward row 0) |
| **Left** | 2 | `[-1, 0]` | Move one cell left |
| **Down** | 3 | `[0, +1]` | Move one cell down (toward higher rows) |
| **Noop** | 4 | `[0, 0]` | Stay in place |

### Movement Calculation

```python
new_position = current_position + direction_vector

# Example: Agent at [2, 2] takes action Right (0)
# new_position = [2, 2] + [1, 0] = [3, 2]
```

!!! note "Y-Axis Direction" Action Up (index 1) moves toward lower y-values because the origin [0, 0] is at the top-left, not bottom-left.

## Step logic

The [`step()`](api/gridworld.md#step) method is the heart of the environment. It processes all agent actions and advances the simulation by one timestep.

### Order of Operations 

```text
__________________________________________________________________
|                    step(actions) Pipeline                      |
|________________________________________________________________|
|                                                                |
|  1. RECEIVE ACTIONS                                            |
|     actions = {"P1": 0, "P2": 1, "R1": 2, "R2": 3}             |
|                                                                |
|  2. PROCESS MOVES (for each agent)                             |
|     |-- Skip captured agents                                   |
|     |-- Calculate new position                                 |
|     |-- Validate move (bounds + obstacles)                     |
|                                                                |
|  3. UPDATE POSITIONS                                           |
|     All valid moves are applied simultaneously                 |
|                                                                |
|  4. DETECT CAPTURES                                            |
|     Check if any predator shares cell with prey                |
|                                                                |
|  5. CALCULATE REWARDS                                          |
|     |-- Base rewards (capture, survival, timestep)             |
|     |-- Shaping rewards (distance-based, optional)             |
|                                                                |
|  6. CHECK TERMINATION                                          |
|     Episode ends when all prey are captured                    |
|                                                                |
|  7. GENERATE OBSERVATIONS                                      |
|     Each agent receives updated observation                    |
|                                                                |
|  8. RETURN RESULTS                                             |
|     {obs, reward, done, truncated, info}                       |
|________________________________________________________________|
```

### Simultaneous Execution

All agents move simultaneously within a single timestep:

```text
Time t:
  P1 at [1, 1], R1 at [3, 1]

Actions:
  P1: Right (0)
  R1: Left (2)

Time t+1:
  P1 at [2, 1]  (moved right)
  R1 at [2, 1]  (moved left)

Same cell! -> CAPTURE
```


## Movement validation

Not all moves are valid. The environment checks each move before applying it.


### Validation Rules 

| Check | Invalid Condition | Result |
|-------|-------------------|--------|
| **Bounds** | New position outside grid | Agent stays in place |
| **Obstacles** | New position is an obstacle | Agent stays in place |
| **Captured** | Agent already captured | Agent cannot move |


### Boundary Handling 

```text
Agent at [0, 2] tries Left (action 2):
  new_pos = [0, 2] + [-1, 0] = [-1, 2]
  -1 < 0 → OUT OF BOUNDS
  Agent stays at [0, 2]
```

```text 
Agent at [4, 2] tries Right (action 0) on 5×5 grid:
  new_pos = [4, 2] + [1, 0] = [5, 2]
  5 >= 5 → OUT OF BOUNDS
  Agent stays at [4, 2]
```

### Obstacle Collision

```text
Grid:
 ___________________
| P | █ |   |   |   |   P = Predator at [0, 0]
|---|---|---|---|---|   █ = Obstacle at [1, 0]
|   |   |   |   |   |
|___|___|___|___|___|


P takes action Right (0):
  new_pos = [0, 0] + [1, 0] = [1, 0]
  [1, 0] is an obstacle -> BLOCKED
  P stays at [0, 0]
```


## Collision Rules 

### Agent-Agent Collisions

When two agents end up on the same cell, the outcome depends on their types:

| Scenario | Outcome |
|----------|---------|
| **Predator + Prey** | Capture! Prey is removed from play |
| **Predator + Predator** | Both occupy same cell (no conflict) |
| **Prey + Prey** | Both occupy same cell (no conflict) |

### Capture Mechanics

```text
Before:                                 After:

 _____________                           _____________
|   | P |   | R |                       |   |   | P |   |
|___|___|___|___|                       |___|___|___|___|
      ^       ^                                   ^
      |       |                                   |
    [1,0]   [3,0]                               [2,0]

P action: Right (0)                     P moved to [2,0]
R action: Left (2)                      R moved to [2,0]

                                        SAME CELL! -> CAPTURE
                                        R is added to captured_agents
                                        R can no longer move
```

### Multiple Captures
If multiple predators catch multiple prey in the same step, all captures are processed:

```text
P1 catches R1 at [2, 2]  → +1 capture
P2 catches R2 at [4, 1]  → +1 capture

captures_this_step = 2
captures_total += 2
```

## Obstacles

Obstacles are impassable cells that block agent movement.

### Configuration

```python
env = GridWorldEnv(
    agents=agents,
    size=8,
    perc_num_obstacle=20.0,  # 20% of cells are obstacles
)

# For 8×8 grid: 64 cells × 20% = ~12 obstacles
```

### Placement Rules 

1. Obstacles are placed randomly during [`reset()`](api/gridworld.md#reset).
2. Obstacles cannot be placed on agent starting positions
3. Obstacle positions are stored in [`_obstacle_location()`](api/gridworld.md#_obstable_location)


### Visual Example

```text
8x8 Grid with 20% obstacles (~12 obstacles):

 _______________________________
| P1|   | █ |   | █ |   |   |   |
|---|---|---|---|---|---|---|---|
|   | █ |   |   |   | █ |   | R1|
|---|---|---|---|---|---|---|---|
|   |   |   | █ |   |   | █ |   |
|---|---|---|---|---|---|---|---|
| █ |   |   |   |   | █ |   |   |
|---|---|---|---|---|---|---|---|
|   |   | █ |   |   |   |   | █ |
|---|---|---|---|---|---|---|---|
|   | P2|   |   | █ |   |   |   |
|---|---|---|---|---|---|---|---|
| █ |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|   |   |   | R2|   |   |   |   |
|___|___|___|___|___|___|___|___|

P = Predator, R = Prey, █ = Obstacle
```

## Rewards 
The environment provides two types of rewards:

### Base Rewards 
The environment provides two types of rewards:

| Agent Type | Event | Reward | Purpose |
|------------|-------|--------|---------|
| **Predator** | Captures prey | `+10.0` | Main objective |
| **Predator** | Each timestep | `-0.01` | Encourages faster capture |
| **Prey** | Survives timestep | `+0.1` | Main objective |
| **Prey** | Gets captured | `-10.0` | Penalty for failure |

### Reward Shaping (Optional)
Distance-based shaping rewards provide denser feedback:

| Agent Type | Condition | Shaping Reward |
|------------|-----------|----------------|
| **Predator** | Moves closer to prey | Positive |
| **Predator** | Moves away from prey | Negative |
| **Prey** | Moves away from predators | Positive |
| **Prey** | Moves closer to predators | Negative |

!!! info "Why Reward Shaping?" Without shaping, predators only receive reward upon capture (+10). This "sparse reward" makes learning difficult because the agent gets no feedback until success.

```text 
Shaping rewards provide continuous guidance: "You're getting warmer!"
```

### Total Reward Calculation

```python 
total_reward = base_reward + shaping_reward
```

### Example Episode Rewards 
```text
Step 1:
  P1: base=-0.01, shaping=+0.3 (moved closer) → total=+0.29
  R1: base=+0.1,  shaping=+0.2 (moved away)   → total=+0.30

Step 2:
  P1: base=-0.01, shaping=-0.2 (moved away)   → total=-0.21
  R1: base=+0.1,  shaping=-0.1 (moved closer) → total=+0.00

Step 3: (P1 catches R1)
  P1: base=+10.0, shaping=+0.5               → total=+10.50
  R1: base=-10.0, shaping=0                  → total=-10.00
```


## Episode Lifecycle

### Starting an Episode

```python
# Reset creates fresh grid with new obstacle/agent positions
obs, info = env.reset(seed=42)
```

### Running the Episode
```python 
done = False
while not done:
    # Get actions from policies
    actions = {"P1": 0, "R1": 1}
    
    # Execute step
    result = env.step(actions)
    
    obs = result["obs"]
    rewards = result["reward"]
    done = result["done"]
    info = result["info"]
```

### Termination Conditions 

| Condition | `done` | `truncated` |
|-----------|--------|-------------|
| All prey captured | `True` | `False` |
| Max steps reached | `False` | `True` |
| Custom condition | Configurable | Configurable |

### Cleanup

```python
env.close()  # Release pygame resources
```


## Observations 
Each agent receives an observation containing:

### Local Observation

```python 
obs["P1"]["local"]  # Agent's own position
# array([2, 3])
```

### Global Observation

```python 
obs["P1"]["global"]["other_agents"]
# [
#     {"name": "R1", "type": "prey", "position": [5, 2], "distance": 3.16},
#     {"name": "P2", "type": "predator", "position": [1, 4], "distance": 1.41},
# ]

obs["P1"]["global"]["obstacles"]
# [array([3, 0]), array([1, 2]), ...]
```

### Observation Structure 

```python
observations = {
    "P1": {
        "local": array([2, 3]),
        "global": {
            "other_agents": [...],
            "obstacles": [...]
        }
    },
    "R1": {
        "local": array([5, 2]),
        "global": {
            "other_agents": [...],
            "obstacles": [...]
        }
    }
}
```

## Rendering 
The environment supports pygame-based visualization.

### Render Modes 

| Mode | Description | Use Case |
|------|-------------|----------|
| `"human"` | Opens pygame window | Watching/debugging |
| `"rgb_array"` | Returns pixel array | Recording videos |
| `None` | No rendering | Fast training |

### Enabling Rendering 

```python 
# Option 1: At creation
env = GridWorldEnv(agents=agents, render_mode="human")

# Option 2: Manual render calls
env.render()
```

### Visual Elements 

| Element | Appearance |
|---------|------------|
| **Grid lines** | Light gray |
| **Obstacles** | Dark gray squares |
| **Predators** | Red shapes (circle, square, etc.) |
| **Prey** | Green shapes |
| **Other agents** | Blue shapes |
| **Labels** | Agent name on each shape |

## Configureation summary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| [`agents`](api/gridworld.md#agents) | `List[Agent]` | **Required** | Agents in the environment |
| [`size`](api/gridworld.md#size) | `int` | `5` | Grid dimensions (size × size) |
| [`perc_num_obstacle`](api/gridworld.md#perc_num_obstacle) | `float` | `30.0` | Percentage of obstacle cells |
| [`render_mode`](api/gridworld.md#render_mode) | `str` | `None` | Rendering mode |
| [`window_size`](api/gridworld.md#window_size) | `int` | `600` | Pygame window size (pixels) |
| [`seed`](api/gridworld.md#seed) | `int` | `None` | Random seed for reproducibility |


## Quick Reference 

### Creating Environment

```python 
from multi_agent_package.agents import Agent
from multi_agent_package.gridworld import GridWorldEnv

agents = [
    Agent("predator", "predator_1", "P1"),
    Agent("prey", "prey_1", "R1"),
]

env = GridWorldEnv(
    agents=agents,
    size=8,
    perc_num_obstacle=20.0,
    render_mode="human",
    seed=42
)
```

### Running Episode

```python
obs, info = env.reset()
done = False

while not done:
    actions = {"P1": 0, "R1": 1}
    result = env.step(actions)
    done = result["done"]

env.close()
```

### Step Return Value

```python
result = env.step(actions)

result["obs"]        # Dict[str, Dict] - observations
result["reward"]     # Dict[str, float] - rewards
result["done"]       # bool - episode finished?
result["truncated"]  # bool - episode cut short?
result["info"]       # Dict - additional info
```