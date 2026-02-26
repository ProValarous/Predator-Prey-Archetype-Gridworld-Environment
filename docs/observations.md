# Observations Module

## Overview

The Observations module defines how agents perceive the environment state at each timestep. Observations are constructed via classes that inherit **ObservationBuilder** class as the base abstract class. The desired class is selected at runtime through YAML configuration and a registry pattern.

The module supports five observation modes:

| Mode | Observability | Frame | Description |
|------|---------------|-------|-------------|
| **Default** | Full | Mixed | Scalar distances to all agents and obstacles |
| **LocalOnly** | Minimal | Absolute | Agent's own position only |
| **LocalRadius** | Partial | Hybrid | Entities within Manhattan radius (relative positions) |
| **Absolute** | Full | World | All positions in grid coordinates |
| **Relative** | Full | Egocentric | All positions relative to agent (agent at origin) |

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
    "absolute": AbsoluteObservation,
    "relative": RelativeObservation,
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


## AbsoluteObservation
Location: multi_agent_package/observations/absolute.py

Purpose: Provides full observability with world coordinates. All entity positions are expressed in global grid coordinates.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `include_agents` | `bool` | `True` | Include other agents |
| `include_obstacles` | `bool` | `True` | Include obstacles |
| `distance_type` | `str` | `"euclidean"` | `"euclidean"` or `"manhattan"` |

### Output Structure 
```python
{
    "agent_name": {
        "local": {
            "pos": np.array([x, y]),      # ABSOLUTE world position
            "type": "predator",
            "team": "predator_1",
            "speed": 1,
        },
        "agents": {
            "other_agent_name": {
                "pos": np.array([x, y]),  # ABSOLUTE world position
                "dist": float,             # Distance to agent
                "type": "prey",
                "team": "prey_1",
            }
        },
        "obstacles": {
            "obstacle_0": {
                "pos": np.array([x, y]),  # ABSOLUTE world position
                "dist": float,
            }
        }
    }
}
```
### Example
```text
Grid (7x7):
- Predator P at [2, 3]
- Prey R at [5, 1]
- Obstacle at [3, 3]

Output for P:
{
    "predator_1": {
        "local": {
            "pos": np.array([2, 3]),     # P's world position
            "type": "predator",
            "speed": 1,
        },
        "agents": {
            "prey_1": {
                "pos": np.array([5, 1]), # R's world position
                "dist": 5.0,
                "type": "prey",
            }
        },
        "obstacles": {
            "obstacle_0": {
                "pos": np.array([3, 3]), # Obstacle world position
                "dist": 1.0,
            }
        }
    }
}
```

## RelativeObservation

Location: multi_agent_package/observations/relative.py

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `include_agents` | `bool` | `True` | Include other agents |
| `include_obstacles` | `bool` | `True` | Include obstacles |
| `include_walls` | `bool` | `False` | Include wall distances |
| `distance_type` | `str` | `"manhattan"` | `"euclidean"` or `"manhattan"` |

### Output Structure
```python
{
    "agent_name": {
        "local": {
            "pos": np.array([0, 0]),      # ALWAYS origin
            "type": "predator",
            "team": "predator_1",
            "speed": 1,
        },
        "agents": {
            "other_agent_name": {
                "rel_pos": np.array([dx, dy]),  # RELATIVE offset
                "dist": int,                     # Distance
                "type": "prey",
                "team": "prey_1",
            }
        },
        "obstacles": {
            "obstacle_0": {
                "rel_pos": np.array([dx, dy]),  # RELATIVE offset
                "dist": int,
            }
        },
        "walls": {                              # If include_walls=True
            "left": int,
            "right": int,
            "up": int,
            "down": int,
        }
    }
}
```

### Example

```text
Grid (7x7):
- Predator P at [2, 3]
- Prey R at [5, 1]
- Obstacle at [3, 3]

Output for P:
{
    "predator_1": {
        "local": {
            "pos": np.array([0, 0]),          # P is always at origin
            "type": "predator",
            "speed": 1,
        },
        "agents": {
            "prey_1": {
                "rel_pos": np.array([3, -2]), # R relative to P: [5,1] - [2,3]
                "dist": 5,                     # |3| + |-2| = 5 (manhattan)
                "type": "prey",
            }
        },
        "obstacles": {
            "obstacle_0": {
                "rel_pos": np.array([1, 0]),  # Obstacle relative to P
                "dist": 1,
            }
        },
        "walls": {
            "left": 2,                        # P is 2 cells from left edge
            "right": 4,                       # P is 4 cells from right edge
            "up": 3,                          # P is 3 cells from top
            "down": 3,                        # P is 3 cells from bottom
        }
    }
}
```


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