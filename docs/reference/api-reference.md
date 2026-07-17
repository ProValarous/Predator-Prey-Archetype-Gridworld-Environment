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
    allow_cell_sharing:       bool           = True,   # accepted and stored, but never read anywhere — currently dead
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
env.observation_encoder  = callable(obs, env) -> array-like   # required by DQN only
env.reward_fn            = callable(env) -> Dict[str, float]
env.action_space_plugin  = ActionSpace   # object with .to_direction(int) -> np.ndarray
```

> Note: `render_mode="rgb_array"` is accepted but currently non-functional — both `render()` and `_render_frame()` return `None` for any mode other than `"human"`, so the pixel-array-returning code path is unreachable dead code.

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
| `agent_speed` | `int` | Defaults to 1 (predator)/3 (prey) from `__init__`, but `build_agents()` always overwrites it from `agents.yaml`. Ignored by `GridWorldEnv.step()` itself; consumed by `SpeedWrapper` for sub-step budgeting. |
| `stamina` | `int` | Default 10. Ignored by `GridWorldEnv.step()` itself; depleted by `SpeedWrapper` (1 per sub-step), reset to max on `env.reset()`. |
| `_agent_location` | `np.ndarray (2,)` | Current `[x, y]` position |
| `action_space` | `gym.spaces.Discrete(5)` | 5 discrete actions |

**Actions (default `discrete_5`; see [reference/config-reference.md](config-reference.md#actionsyaml) for `cross` and `speed_discrete_5`):**

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

## DQN

```python
DQN(env, config: dict)
# config keys: gamma, epsilon, epsilon_decay, min_epsilon, episodes, seed,
#              batch_size, buffer_size, min_replay_size, target_update_interval,
#              learning_rate, hidden_layers, grad_clip, device, verbose,
#              log_interval, debug_first_episode, save_path, curves_path,
#              double_dqn (bool, default False), dueling (bool, default False)
# action_dim is NOT read as a plain default — resolved via _resolve_action_dim():
#   env.action_space_plugin.n_actions if set, else env.action_space.n;
#   raises ValueError if config also sets action_dim and it disagrees

# Precondition: env.observation_encoder must already be a callable(obs, env) -> array-like
# (run_from_config.build_environment() attaches this from the observation builder's encode())

algo.q_networks           # Dict[str, QNetwork | DuelingQNetwork]   — one per agent
algo.target_networks      # Dict[str, QNetwork | DuelingQNetwork]   — hard-synced every target_update_interval
algo.replay_buffers       # Dict[str, ReplayBuffer]                 — one per agent, seed+i per agent
algo.state_dim            # int   — derived from encoding the first agent's initial observation
algo.action_dim           # int   — resolved as above
algo._resolve_action_dim(config: dict) -> int
algo._encode_observation(obs) -> np.ndarray   # flattened float32 vector via observation_encoder
```

```python
# src/baselines/DQN/q_network.py
QNetwork(input_dim, hidden_layers, output_dim)          # plain MLP, linear output (Q-values can be negative)
DuelingQNetwork(input_dim, hidden_layers, output_dim)   # value_head + advantage_head, recombined:
                                                          # Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

# src/baselines/DQN/replay_buffer.py
ReplayBuffer(capacity: int, state_dim: int, seed=None)
buffer.push(state, action, reward, next_state, done)
buffer.sample(batch_size) -> (states, actions, rewards, next_states, dones)  # without replacement
len(buffer)   # current size (<= capacity)
```

---

## Wrappers

```python
# src/multi_agent_package/wrappers/speed.py
SpeedWrapper(env)   # wraps a fully-wired GridWorldEnv; apply LAST in build_environment()

wrapper.step(actions: Dict[str, int]) -> dict     # same {"obs","reward","terminated","truncated","info"} shape
wrapper.reset(**kwargs)                           # resets stamina to max, delegates to env.reset()
wrapper.close()
wrapper.NOOP                                       # class constant, 4
# Unknown attributes (env.agents, env.action_space_plugin, env.observation_encoder, ...)
# proxy through to the wrapped env via __getattr__.
```

---

## Registries

```python
# Observation registry
from multi_agent_package.registry.observation_registry import (
    get_observation_builder,      # (name: str, **params) -> ObservationBuilder
    register_observation,         # (name: str, cls: Type[ObservationBuilder]) -> None  (validates issubclass)
)

# Reward registry
from multi_agent_package.registry.reward_registry import (
    get_reward_function,          # (name: str, weight: float = 1.0, **params) -> RewardFunction
    register_reward,              # (name: str, cls: Type[RewardFunction]) -> None  (validates issubclass)
)

# Action registry
from multi_agent_package.registry.action_registry import (
    get_action_space,             # (name: str, **params) -> ActionSpace
    register_action_space,        # (name: str, cls: Type[ActionSpace]) -> None  (does NOT validate issubclass)
)

# Algorithm registry
from baselines.registry.algorithm_registry import (
    get,                          # (name: str) -> Type[BaseAlgorithm]
    register,                     # (name: str, cls) -> None  (raises if name already registered)
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
#           "rewards": dict, "actions": dict, "experiment": dict}

build_agents(agent_cfg: dict) -> List[Agent]

build_environment(configs: dict) -> SpeedWrapper   # NOT a raw GridWorldEnv
# Wires, in order: GridWorldEnv construction, observation_builder +
# observation_encoder, reward_fn (base_reward() is NOT re-added — it's
# already unconditional inside step()), action_space_plugin, then wraps
# the result in SpeedWrapper (must be last, since it proxies the above
# via __getattr__).

main(config_dir: str = "configs") -> None
# load_all_configs -> build_environment -> get_algorithm(name) -> algo.train() -> env.close()
```

Each algorithm also has a thin per-algorithm entry point under `scripts/` (`run_iql.py`, `run_cql.py`, `run_mixed.py`, `run_dqn.py`), all sharing the same CLI shape:

```
--mode {train,eval}   (default: train)
--config-dir DIR      (default: "configs")
--save-path PATH      (default: "trained_<algo>.pkl")
--load-path PATH      (required for --mode eval)
--render              (store_true; only affects --mode eval)
```
