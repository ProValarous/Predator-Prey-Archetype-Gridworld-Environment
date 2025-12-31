# Agent Abstractions

## Overview

Agents in the Predator-Prey GridWorld are entities that can observe, decide,
and act within the environment. Each agent has a type, team, and unique name.

This page explains the concepts. For method details, see the
[API Reference](api/agents.md).

---

## Agent Roles

The environment supports three agent types with different behaviors:

| Role | Speed | Color | Objective |
|------|-------|-------|-----------|
| **Predator** | 1 | Red | Capture prey by occupying the same cell |
| **Prey** | 3 | Green | Survive by evading predators |
| **Other** | 1 | Blue | Custom behavior (user-defined) |

!!! note "Speed Asymmetry"
    Prey move 3x faster than predators. This asymmetry means predators
    must **coordinate** to corner and capture prey, rather than simply
    chasing them.

---

## Creating Agents

### Basic Example

```python
from multi_agent_package.agents import Agent

# Create a predator
predator = Agent(
    agent_type="predator",
    agent_team="predator_1",
    agent_name="Hunter"
)

# Create a prey
prey = Agent(
    agent_type="prey",
    agent_team="prey_1",
    agent_name="Runner"
)

# Check properties
print(predator.agent_speed)  # Output: 1
print(prey.agent_speed)      # Output: 3
```

## Team Identifier Formats
The agent_team parameter accepts multiple formats:
```python
# Integer format
agent1 = Agent("predator", 1, "P1")

# String with underscore: "type_subteamID"
agent2 = Agent("predator", "predator_2", "P2")

# Numeric string
agent3 = Agent("prey", "3", "R3")
```

| Format | Example | Parsed As |
|--------|---------|-----------|
| Integer | `3` | Subteam 3 |
| String | `"predator_2"` | Type: predator, Subteam: 2 |
| Numeric string | `"2"` | Subteam 2 |

## Action Space

Each agent can perform 5 discrete actions:

| Action | Code | Direction Vector |
| :--- | :--- | :--- |
| Right | 0 | [1, 0] |
| Up | 1 | [0, 1] |
| Left | 2 | [-1, 0] |
| Down | 3 | [0, -1] |
| Noop | 4 | [0, 0] |

```python 
# Access action space
print(agent.action_space)  # Discrete(5)

# Get direction for an action
direction = agent._actions_to_directions[0]  # array([1, 0]) = Right
```

### Action Diagram

```text
      Up (1)
        ^
        |
Left (2) <---> Right (0)
        |
        v
      Down (3)

Noop (4) = Stay in place
```


## Agent Properties 

### Identity Properties 

| Property | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `agent_type` | `str` | Role of the agent | `"predator"` |
| `agent_team` | `str` or  `int` | Team/subteam identifier | `"predator_1"` |
| `agent_name` | `str` | Unique display name | `"Hunter"` |

### Gameplay properties 

| Property | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `agent_speed` | `int` | Movement multiplier | 1 or 3 |
| `stamina` | `int` | Energy resource | 10 |
| `_agent_location` | `np.ndarray` | Current `[x, y]` position | `[0, 0]` |

## Getting Agent information

```python 
# Get current observation
obs = agent._get_obs()
print(obs["local"])  # array([x, y])

# Get agent metadata
info = agent._get_info()
print(info)
# {
#     "name": "Hunter",
#     "type": "predator",
#     "team": "predator_1",
#     "speed": 1,
#     "stamina": 10
# }
```

### Rendering 
Agents are rendered with distinct colors and shapes for visual identification.

### Colors by Type 
| Agent Type | Base Hue | Color Family |
| :--- | :--- | :--- |
| Predator | 0° | Reds |
| Prey | 120° | Greens |
| Other | 240° | Blues |

Subteams within the same type get different saturation/brightness levels.

```python 
# Get agent's RGB color
r, g, b = agent.get_agent_color()
print(f"RGB: ({r}, {g}, {b})")
```

### Shapes by Subteam

Shapes cycle based on subteam ID:
| Subteam ID | Shape |
| :--- | :--- |
| 1 | Circle |
| 2 | Square |
| 3 | Triangle |
| 4 | Star |
| 5 | Diamond |
| 6+ | Cycles back to Circle |

## Multi-Agent Setup 
### Creating Multiple Agents 
```python 
from multi_agent_package.agents import Agent

# Create predator team
predators = [
    Agent("predator", "predator_1", "Hunter1"),
    Agent("predator", "predator_2", "Hunter2"),
]

# Create prey team
prey = [
    Agent("prey", "prey_1", "Runner1"),
    Agent("prey", "prey_2", "Runner2"),
]

# Combine for environment
all_agents = predators + prey
```

### 2v2 Configuration example 
```python 
from multi_agent_package.agents import Agent
from multi_agent_package.gridworld import GridWorldEnv

# Create agents
agents = [
    Agent("predator", "predator_1", "P1"),
    Agent("predator", "predator_2", "P2"),
    Agent("prey", "prey_1", "R1"),
    Agent("prey", "prey_2", "R2"),
]

# Create environment
env = GridWorldEnv(agents=agents, size=10, render_mode="human")

# Run simulation
obs, info = env.reset()
for _ in range(100):
    actions = {agent.agent_name: agent.action_space.sample() for agent in agents}
    obs, rewards, done, truncated, info = env.step(actions)
    if done:
        break
```

---

## Summary

| Concept | Key Points |
|---------|------------|
| **Types** | predator (slow), prey (fast), other (custom) |
| **Teams** | Used for color/shape differentiation |
| **Actions** | 5 discrete: Right, Up, Left, Down, Noop |
| **Observations** | Local position + optional global state |
| **Rendering** | Color by type, shape by subteam |

---

## API Reference

For complete method documentation, see [Agent API Reference](api/agents.md).