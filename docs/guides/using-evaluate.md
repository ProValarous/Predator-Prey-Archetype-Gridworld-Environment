# Guide: Collecting Metrics with evaluate.py

`src/multi_agent_package/scripts/evaluate.py` runs an algorithm against the environment for a number of episodes and returns simple aggregate metrics.

> ⚠️ **This does not load a trained checkpoint.** `evaluate()` builds a **fresh** algorithm instance from config (`algo_cls(env, algo_cfg.get("params", {}))`) — there is no `load_path` parameter and no `.load(...)` call anywhere in this file. Calling it as-is evaluates an untrained (or however-initialized) policy, not a saved one. If you want to evaluate a saved checkpoint, use each algorithm's own `run_<algo>.py --mode eval --load-path ...` script instead (see [guides/quickstart.md](quickstart.md)), or load the checkpoint yourself and write a short loop (see [Writing your own eval loop](#writing-your-own-eval-loop) below).

---

## Actual signature

```python
def evaluate(config_dir: str = "configs", episodes: int = 10) -> dict:
```

It internally calls `load_all_configs(config_dir)`, forces `render_mode = None` (headless), builds the environment via `build_environment()`, looks up the algorithm class via `get_algorithm(algo_cfg["name"])`, constructs a fresh instance, then runs `episodes` full episodes calling `algo.select_actions(obs)` each step — **no epsilon override**, no training update. Whatever `epsilon` the algorithm's config specifies is what it uses (many experiment YAMLs default to a high initial `epsilon`, which means this is *not* automatically a greedy/exploitation-only rollout unless you set `epsilon` low in config first).

## Usage

```python
from multi_agent_package.scripts.evaluate import evaluate

results = evaluate(config_dir="configs", episodes=20)
print(results)
```

**Actual output shape:**
```python
{
    "mean_episode_length": 47.2,
    "std_episode_length": 8.1,
    "mean_return_pred_1": -12.4,
    "mean_return_prey_1": 3.7,
    # one "mean_return_<agent_id>" key per agent, dynamically
}
```

There is **no** `episode_lengths` list and **no** `capture_rate` key — only the two length-summary stats and one mean-return key per agent.

## From the CLI

```bash
PYTHONPATH=src python -m multi_agent_package.scripts.evaluate
```

There are **no CLI flags** — the `if __name__ == "__main__":` block simply calls `evaluate("configs", episodes=10)` with hardcoded arguments and prints the summary. To evaluate a different config directory or episode count, either edit the script or call `evaluate()` from your own code.

## Running with a display (visual eval)

`evaluate()` always forces `render_mode = None`. To watch episodes visually, don't use `evaluate()` — build the environment yourself with `render_mode="human"` and write your own loop (below), or use `run_<algo>.py --mode eval --render`.

---

## Writing your own eval loop

To evaluate an actual saved checkpoint, or collect metrics `evaluate()` doesn't (per-step Q-value norms, capture positions, capture rate):

```python
from multi_agent_package.scripts.run_from_config import load_all_configs, build_environment
from baselines.IQL.iql import IQL
import baselines  # noqa: F401 — triggers auto-registration

configs = load_all_configs(experiment_file="experiment_iql.yaml")
configs["env"]["env"]["render_mode"] = None
env = build_environment(configs)

params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
algo = IQL.load(env, params, "my_trained_iql.pkl")   # actually loads saved weights

episodes = 20
captures = 0
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    ep_return = {aid: 0.0 for aid in obs}
    while not done:
        actions = algo.select_actions(obs)
        out = env.step(actions)
        for aid in obs:
            ep_return[aid] += out["reward"][aid]
        obs = out["obs"]
        done = out["terminated"] or out["truncated"]
    if out["terminated"]:
        captures += 1
    # log ep_return, env._captures_total, env._episode_steps, etc.

print(f"capture_rate = {captures / episodes:.2f}")
```

> **Note:** Reading `env._captures_total` or `env._episode_steps` directly is acceptable in eval scripts and notebooks. The "black-box env" rule (see [specs/algorithm-spec.md](../specs/algorithm-spec.md)) applies to learning algorithms that must generalize across env variants, not to one-off analysis code.
