# API Reference

Key method signatures and return types for the stable public interfaces.

---

## GridWorldEnv

```python
GridWorldEnv(
    agents:                   List[Agent],
    render_mode:              Optional[str]  = None,        # "human" | "rgb_array" | None
    size:                     int            = 5,
    perc_num_obstacle:        float          = 30.0,        # % of cells blocked
    window_size:              int            = 600,
    seed:                     Optional[int]  = None,
    capture_threshold:        int            = 1,
    max_steps:                Optional[int]  = None,
    allow_cell_sharing:       bool           = True,
    block_agents_by_obstacles: bool          = True,
)
```

```python
env.reset(
    seed:    Optional[int]  = None,
    options: Optional[dict] = None,
) -> Tuple[Dict[str, dict], Dict[str, dict]]
#            obs               info
```

```python
env.step(
    action: Dict[str, int]   # {agent_name: action_int}
) -> Dict[str, object]
# {
#   "obs":        Dict[str, dict],
#   "reward":     Dict[str, float],
#   "terminated": bool,
#   "truncated":  bool,
#   "info":       Dict[str, dict],
# }
```

```python
env.base_reward() -> Dict[str, float]
# Called internally by step(). Safe to call in reward functions and eval scripts.
# Returns the hardcoded reward signals (capture, step cost, obstacle penalty).
```

```python
env.close() -> None
# Shuts down pygame window if open.
```

**Extension hooks (set before training):**
```python
env.observation_builder  = callable(env) -> Dict[str, dict]
env.reward_fn            = callable(env) -> Dict[str, float]
env.action_space_plugin  = ActionSpace   # object with .to_direction(int) -> np.ndarray
```

---

## Agent

```python
Agent(
    agent_type: str,    # "predator" | "prey" | any string
    agent_team: any,    # str or int, e.g. "predator_1" or 1
    agent_name: str,    # unique identifier, e.g. "pred_1"
)
```

**Key attributes (read-only):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `agent_name` | `str` | Unique identifier used as dict key |
| `agent_type` | `str` | `"predator"` or `"prey"` |
| `agent_team` | `str\|int` | Team identifier |
| `agent_speed` | `int` | 1 for predators, 3 for prey — stored only, not used in step |
| `stamina` | `int` | Default 10 — stored only, not used in any mechanic |
| `_agent_location` | `np.ndarray (2,)` | Current `[x, y]` position |
| `action_space` | `gym.spaces.Discrete(5)` | 5 discrete actions |

**Actions:**

| Int | Direction | Grid delta |
|-----|-----------|-----------|
| 0 | Right | `[+1, 0]` |
| 1 | Up (visual down) | `[0, +1]` |
| 2 | Left | `[-1, 0]` |
| 3 | Down (visual up) | `[0, -1]` |
| 4 | Noop | `[0, 0]` |

> Y increases downward in screen space. Action 1 moves toward higher Y values, which appears downward on screen.

```python
agent._get_obs(global_obs: Optional[dict] = None) -> dict
# Returns {"local": np.ndarray([x,y]), "global": global_obs}

agent._get_info() -> dict
# Returns {"name": str, "type": str, "team": any, "speed": int, "stamina": int}
```

---

## BaseAlgorithm

```python
class MyAlgorithm(BaseAlgorithm):
    def __init__(self, env, config: dict): ...
    def select_actions(self, observations: Dict[str, dict]) -> Dict[str, int]: ...
    def train(self) -> None: ...
    def evaluate(self, episodes: int = 5, max_steps: int = 500) -> None: ...  # inherited
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, env, config: dict, path: str) -> "MyAlgorithm": ...
```

---

## IQL

```python
IQL(env, config: dict)
# config keys: alpha, gamma, epsilon, epsilon_decay, min_epsilon,
#              action_dim (default 5), episodes, seed

algo.q_tables          # Dict[str, defaultdict]  — one table per agent
algo.agent_ids         # List[str]               — agent names from initial reset
algo._encode_state(obs: dict) -> tuple            — hashable state key
```

---

## CQL

```python
CQL(env, config: dict)
# config keys: same as IQL (no predator_algo / prey_algo)

algo.q_table           # defaultdict   — joint_state → np.ndarray(n_joint_actions,)
algo.n_agents          # int
algo.n_joint_actions   # int           — action_dim^n_agents
algo._joint_state(observations: dict) -> tuple
algo._joint_action_index(actions: dict) -> int
```

---

## MixedTrainer

```python
MixedTrainer(env, config: dict)
# config keys: same as IQL + predator_algo: "iql"|"cql", prey_algo: "iql"|"cql"

algo._iql_tables          # Dict[str, defaultdict]   — one per IQL agent
algo._cql_tables          # Dict[str, defaultdict]   — one per CQL team ("predator"|"prey")
algo._predators           # List[str]                — predator agent names
algo._prey                # List[str]                — prey agent names
algo._team_of             # Dict[str, str]           — agent_name → "predator"|"prey"
algo._cql_team_ids        # Dict[str, List[str]]     — team_key → ordered member list
algo._team_joint_state(observations, team_key) -> tuple
algo._team_joint_action_index(actions, team_key) -> int
```

---

## Registries

```python
# Observation registry
from multi_agent_package.registry.observation_registry import (
    get_observation_builder,      # (name: str, **params) -> ObservationBuilder
    register_observation,         # (name: str, cls: Type[ObservationBuilder]) -> None
)

# Reward registry
from multi_agent_package.registry.reward_registry import (
    get_reward_function,          # (name: str, weight: float = 1.0, **params) -> RewardFunction
    register_reward,              # (name: str, cls: Type[RewardFunction]) -> None
)

# Algorithm registry
from baselines.registry.algorithm_registry import (
    get,                          # (name: str) -> Type[BaseAlgorithm]
    register,                     # (name: str, cls) -> None
    list_algorithms,              # () -> List[str]
)
```

---

## run_from_config.py

```python
load_all_configs(
    config_dir:      str = "configs",
    experiment_file: str = "experiment.yaml",
) -> dict
# Returns {"env": dict, "agents": dict, "observations": dict,
#           "rewards": dict, "experiment": dict}

build_agents(agent_cfg: dict) -> List[Agent]

build_environment(configs: dict) -> GridWorldEnv
# Wires observation_builder and reward_fn before returning.
```
