# Guide: Collecting Metrics with evaluate.py

`src/multi_agent_package/scripts/evaluate.py` runs a trained algorithm against the environment and returns structured metrics.

---

## Usage

```python
from multi_agent_package.scripts.evaluate import evaluate
from multi_agent_package.scripts.run_from_config import load_all_configs, build_environment
from baselines.IQL.iql import IQL
import baselines  # triggers auto-registration

configs = load_all_configs(experiment_file="experiment_iql.yaml")
configs["env"]["env"]["render_mode"] = None   # headless
env = build_environment(configs)

params = configs["experiment"]["experiment"]["algorithm"].get("params", {})
algo = IQL.load(env, params, "my_trained_iql.pkl")

results = evaluate(algo, env, episodes=20)
print(results)
```

**Output:**
```python
{
    "mean_return": {
        "pred_1": -312.4,
        "prey_1": -88.1,
    },
    "episode_lengths": [47, 52, 38, ...],   # steps per episode
    "capture_rate": 0.75,                   # fraction of episodes that ended in capture
}
```

---

## From the CLI

```bash
cd src
python -m multi_agent_package.scripts.evaluate \
    --load-path my_trained_iql.pkl \
    --algorithm iql \
    --episodes 20
```

---

## What evaluate.py measures

| Metric | Description |
|--------|-------------|
| `mean_return[agent]` | Average total reward per agent across all episodes |
| `episode_lengths` | List of step counts per episode |
| `capture_rate` | `terminated` episodes ÷ total episodes |

The eval loop uses `epsilon=0.0` (greedy, no exploration). The environment is fully wired (same observation and reward functions as training).

---

## Running with a display (visual eval)

```python
configs["env"]["env"]["render_mode"] = "human"
env = build_environment(configs)
# ... load algo ...
results = evaluate(algo, env, episodes=5)
```

Requires pygame and a display (not available in headless CI).

---

## Writing your own eval loop

If you need metrics `evaluate.py` doesn't collect (e.g., per-step Q-value norms, capture positions):

```python
algo.epsilon = 0.0    # greedy
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
    # log ep_return, env._captures_total, env._episode_steps, etc.
```

> **Note:** Reading `env._captures_total` or `env._episode_steps` directly is acceptable in eval scripts and notebooks. The "black-box env" rule applies only to learning algorithms that must generalize across env variants.
