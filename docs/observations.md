# Observations Module

## Overview

The Observations module defines how agents perceive the environment state at each timestep. Observations are constructed via classes that inherit **ObservationBuilder** class as the base abstract class. The desired class is selected at runtime through YAML configuration and a registry pattern.

The module supports three observation modes:

| Mode | Observability | Description |
|------|---------------|-------------|
| **Default** | Full | All agents see all other agents and obstacles |
| **LocalOnly** | Minimal | Agents see only their own position |
| **LocalRadius** | Partial | Agents see entities within a Manhattan distance threshold |

Observation builders are **read-only**—they query environment state but do not modify it.

---

## Configuration & Registry

### YAML Configuration

Observations are selected via `configs/observations.yaml`:

```yaml
observations:
  type: local_radius          # Registry key (string)
  params:
    radius: 3                 # Passed to builder as **kwargs
    include_obstacles: true
    include_agents: true
```

### Registry Pattern

The observation_registry.py module maintains a mapping from string identifiers to ObservationBuilder subclasses:

```python
_OBSERVATION_REGISTRY: Dict[str, Type[ObservationBuilder]] = {
    "default": DefaultObservation,
    "local_only": LocalOnlyObservation,
    "local_radius": LocalRadiusObservation,
}
```

### Runtime Binding
At environment initialization (run_from_config.py, lines 80-83):
1. YAML file is parsed to extract type and params
2. get_observation_builder(type, **params) retrieves the class from the registry
3. The class is instantiated with the provided parameters
4. The builder's build method is bound to the environment

---

## Observation Base Interface

ObservationBuilder

Location: multi_agent_package/observations/base.py

Purpose: Abstract base class defining the interface for all observation builders.

### Class: Observation Builder 
A base class providing an interface for inheritance and the build() method.

#### Attributes 
- params: the **kwargs taken as input for the class. 

### Contract
- build() must be deterministic given the same environment state
- build() must not modify environment or agent state
- build() must return observations for all agents in env.agents


## Concrete Observation Implementations 

### Default Observation

Location: multi_agent_package/observations/default.py

Purpose: Provides full observability. Each agent receives distances to all other agents and all obstacles.

#### Inputs
Takes an object of GridWorldEnv class as input

#### Output Structure

```python
{
    "agent_name": {
        "local": np.ndarray,        # Agent's own position [x, y]
        "global": {
            "dist_agents": {
                "other_agent_name": float,  # Euclidean distance
                ...
            },
            "dist_obstacles": {
                "obstacle_0": float,
                ...
            }
        }
    }
}
```

#### Dependencies 
- Delegates to GridWorldEnv._default_observations() method
- Requires Agent._get_obs() to construct final observation dict

#### Assumptions 
- All agents are present in env.agents at build time
- Obstacle positions are static within an episode
- Euclidean distance is used for distance calculations

### LocalOnlyObservation

Location: multi_agent_package/observations/local_only.py

Purpose: Provides minimal observability. Agents receive only their own position with no information about other entities.

#### Inputs 
Takes an object of GridWorldEnv class as input

#### Output Structure
```python
{
    "agent_name": {
        "local": np.ndarray,  # Agent's own position [x, y]
        "global": None        # No global information
    }
}
```

#### dependencies
- Directly accesses Agent._agent_location attribute
- No dependency on GridWorldEnv methods beyond agent list

#### Assumptions 
- Agent position is a 2D numpy array of shape (2,)
- Position is in grid coordinates [x, y]

### LocalRadiusObservation

Location: multi_agent_package/observations/local_radius.py

Purpose: Provides partial observability. Agents perceive only entities within a configurable Manhattan distance radius.

#### Inputs 
Takes an object of GridWorldEnv class as input

#### Parameters 

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius` | `int` | `3` | Manhattan distance threshold |
| `include_agents` | `bool` | `True` | Include other agents in observation |
| `include_obstacles` | `bool` | `True` | Include obstacles in observation |

#### Output Structure 
```python
{
    "agent_name": {
        "local": np.ndarray,           # Agent's own position [x, y]
        "visible_agents": {
            "other_agent_name": {
                "rel_pos": tuple,      # (dx, dy) relative to observer
                "dist": int,           # Manhattan distance
                "type": str            # "predator" or "prey"
            }
        },
        "visible_obstacles": {
            "obstacle_i": {
                "rel_pos": tuple,      # (dx, dy) relative to observer
                "dist": int            # Manhattan distance
            }
        },
        "radius": int                  # Configured radius value
    }
}
```

#### Distance Calculation
Manhattan distance is used:
```python
distance = abs(x1 - x2) + abs(y1 - y2)
```

#### Visibility region for radius=2

```text 
        2   1   0   1   2     (Manhattan distance from center)
      ┌───┬───┬───┬───┬───┐
    2 │   │ ○ │ ○ │ ○ │   │
      ├───┼───┼───┼───┼───┤
    1 │ ○ │ ○ │ ○ │ ○ │ ○ │
      ├───┼───┼───┼───┼───┤
    0 │ ○ │ ○ │ A │ ○ │ ○ │   A = Observer agent
      ├───┼───┼───┼───┼───┤   ○ = Visible cells (d ≤ 2)
    1 │ ○ │ ○ │ ○ │ ○ │ ○ │
      ├───┼───┼───┼───┼───┤
    2 │   │ ○ │ ○ │ ○ │   │
      └───┴───┴───┴───┴───┘
```

#### Dependencies 
- Reads agent._agent_location for all agents
- Reads env._obstacle_location for obstacle positions
- Reads agent.agent_type for type information

#### Assumptions 
- Manhattan distance is appropriate for grid-based movement
- Radius is symmetric in all directions
- Relative positions are computed as (target - observer)

---

## Environment and Agent Dependencies 

Required from GridWorldEnv

| Attribute/Method | Type | Usage |
|------------------|------|-------|
| `env.agents` | `List[Agent]` | Iterate over all agents |
| `env._obstacle_location` | `List[np.ndarray]` | Access obstacle positions |
| `env._default_observations()` | `method` | Used by `DefaultObservation` |
| `env.size` | `int` | Grid dimensions (optional) |

Required from Agent

| Attribute | Type | Usage |
|-----------|------|-------|
| `agent.agent_name` | `str` | Dictionary keys for observations |
| `agent.agent_type` | `str` | Type information in radius observation |
| `agent._agent_location` | `np.ndarray` | Position as `[x, y]` |
| `agent._get_obs()` | `method` | Used by `DefaultObservation` |