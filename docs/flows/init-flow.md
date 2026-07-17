# Flow: Initialization

How the system goes from YAML files on disk to a wired, ready-to-train environment.

---

## Trigger

```bash
python -m multi_agent_package.scripts.run_from_config --config configs/
```

---

## Flow Diagram

```
configs/
  env.yaml
  agents.yaml          load_all_configs(config_dir)
  observations.yaml  ──────────────────────────────► merged_configs dict
  rewards.yaml
  actions.yaml
  experiment.yaml
         │
         ▼
  build_agents(agent_cfg)
  ─────────────────────────────────────────────────
  for each entry in agents.yaml:
    ag = Agent(
      agent_type = "predator" | "prey",
      agent_team = "predator_1" | "prey_1" | ...,
      agent_name = "predator_1" | "prey_1" | ...,
    )
    # speed and stamina set as attributes after construction:
    ag.agent_speed = entry["speed"]
    ag.stamina     = entry["stamina"]
  returns: List[Agent]
         │
         ▼
  GridWorldEnv(
    agents                    = agent_list,
    size                      = env_cfg["size"],
    perc_num_obstacle         = env_cfg["obstacle_percentage"],
    render_mode               = env_cfg["render_mode"],
    window_size               = env_cfg["window_size"],
    seed                      = env_cfg["seed"],
    allow_cell_sharing        = dynamics["allow_cell_sharing"],
    block_agents_by_obstacles = dynamics["block_agents_by_obstacles"],
    capture_threshold         = termination["capture_threshold"],
    max_steps                 = termination["max_steps"],
  )
  → env (reward_fn=None, observation_builder=None, action_space_plugin=None)
         │
         ├── Wire observation ──────────────────────────────────────
         │   obs_cfg = configs["observations"]
         │   builder = get_observation_builder(
         │               obs_cfg["type"],        # e.g. "local_radius"
         │               **obs_cfg["params"]     # e.g. radius=3
         │             )
         │   env.observation_builder = builder.build
         │   env.observation_encoder = builder.encode   # required by DQN
         │
         └── Wire reward ───────────────────────────────────────────
             reward_cfg = configs["rewards"]
             reward_fns = []

             # base_reward() is called internally by gridworld.step() on
             # every step — it is NOT added to reward_fns here. Doing so
             # used to double-count every capture/step-cost/obstacle
             # signal (fixed in PR #26). rewards.yaml's base.enabled key
             # is only read to assert it exists; the value is discarded.
             _ = reward_cfg["rewards"]["base"]["enabled"]

             for r in reward_cfg["rewards"].get("shaping", []):
               reward_fns.append(
                 get_reward_function(r["name"], weight=r.get("weight", 1.0))
               )

             def combined_reward(env):
               total = {ag.agent_name: 0.0 for ag in env.agents}
               for rf in reward_fns:
                 for k, v in rf.compute(env).items():
                   total[k] += v
               return total

             env.reward_fn = combined_reward   # ADDED ON TOP of base_reward() inside step()

         └── Wire action space ─────────────────────────────────────
             action_cfg = configs["actions"]
             space = get_action_space(
                       action_cfg["actions"]["type"],    # e.g. "discrete_5"
                       **action_cfg["actions"].get("params", {})
                     )
             env.action_space_plugin = space
                  │
                  ▼
  env is fully wired: observation_builder ✓  observation_encoder ✓
                       reward_fn ✓  action_space_plugin ✓
                  │
                  ▼
  env = SpeedWrapper(env)
  # Applied LAST — it proxies unknown attributes (agents, action_space_plugin,
  # observation_encoder, ...) through to the inner env via __getattr__, so
  # everything above must already be attached for the proxy to expose it.
  # Note: SpeedWrapper hardcodes its own SpeedDiscreteActionSpace() instance
  # for sub-step counting — it does NOT read env.action_space_plugin.
                  │
                  ▼
  algo_cfg = configs["experiment"]["experiment"]["algorithm"]   # note: double "experiment"
  AlgorithmClass = get(algo_cfg["name"])   # from algorithm_registry
  algorithm = AlgorithmClass(env, algo_cfg.get("params", {}))
                  │
                  ▼
  algorithm.train()
```

---

## State After Init

| Attribute | Value |
|-----------|-------|
| `env.agents` | List of N Agent instances (positions unset) |
| `env.reward_fn` | Combined closure over configured **shaping** fns only — base reward is separate, always-on, computed directly by `gridworld.step()` |
| `env.observation_builder` | Bound method of configured observation builder |
| `env.observation_encoder` | Bound `encode` method of the same builder (required by DQN) |
| `env.action_space_plugin` | Configured `ActionSpace` instance |
| `env.rng` | Seeded `np.random.default_rng(seed)` |
| `env.size` | Grid dimension |
| `env._obstacle_location` | Empty list (set on first `reset()`) |
| `env._captured_agents` | Empty list `List[str]` (cleared on each `reset()`) |

The return value of `build_environment()` is actually a `SpeedWrapper` wrapping the object above, not the raw `GridWorldEnv` — all the attributes in this table are still reachable (via `__getattr__` proxying), plus `wrapper._speeds`, `wrapper._max_stamina`, `wrapper._stamina` derived from each agent's `agent_speed`/`stamina`.

The environment is **not ready to step** until `env.reset()` is called. `reset()` places obstacles and agents.

---

## Error Modes

| Symptom | Likely Cause |
|---------|-------------|
| `KeyError: 'local_raidus'` | Typo in `observations.yaml` type field |
| `KeyError: 'discrete_X'` | Unknown key in `actions.yaml` type field; check `action_registry.py` |
| `TypeError: 'NoneType' is not callable` on `step()` | `reward_fn` or `observation_builder` not wired; `reset()` called before wiring |
| `ValueError: Algorithm 'X' not registered` | `import baselines` missing before `get_algorithm()`; auto-registration never ran |
| `KeyError: 'algorithm'` on algo lookup | `load_all_configs()` stores the parsed YAML file's own top-level `experiment:` key under `configs["experiment"]`, so the real path is the double-nested `configs["experiment"]["experiment"]["algorithm"]` — this is correct, verified against `run_from_config.py`'s actual `main()`, not a typo to avoid. |
