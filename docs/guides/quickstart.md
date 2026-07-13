# Quickstart: Checkout to Trained Model

Five steps from a fresh clone to a trained agent.

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

> `pip install -e .` does **not** work in this repo — no build backend is configured, so an editable install won't make `multi_agent_package` importable even if `pip` reports success. Every command below uses `PYTHONPATH=src` instead; run them from the repository root, not `src/`.

Verify the install:

```bash
PYTHONPATH=src python -c "from multi_agent_package.core.gridworld import GridWorldEnv; print('OK')"
```

---

## 2. Configure your experiment

Six YAML files in `configs/` control everything. For a first run, the defaults work:

| File | Controls |
|------|---------|
| `env.yaml` | Grid size, obstacles, seed, termination rules |
| `agents.yaml` | How many predators and prey, their speed/stamina |
| `observations.yaml` | What each agent can see |
| `rewards.yaml` | Reward signals and shaping |
| `actions.yaml` | Which action space (`discrete_5`, `cross`, or `speed_discrete_5`) |
| `experiment.yaml` (or `experiment_{iql,cql,mixed,dqn}.yaml`) | Which algorithm, its hyperparameters |

Each algorithm has its own dedicated runner script that reads the matching experiment file and supports `--mode train|eval`:

```bash
# IQL — reads configs/experiment_iql.yaml
PYTHONPATH=src python -m multi_agent_package.scripts.run_iql

# CQL — reads configs/experiment_cql.yaml
PYTHONPATH=src python -m multi_agent_package.scripts.run_cql

# MixedTrainer — reads configs/experiment_mixed.yaml
PYTHONPATH=src python -m multi_agent_package.scripts.run_mixed

# DQN — reads configs/experiment_dqn.yaml
PYTHONPATH=src python -m multi_agent_package.scripts.run_dqn

# Or a ready-made DQN experiment set (1 predator vs 1 prey, double+dueling enabled)
PYTHONPATH=src python -m multi_agent_package.scripts.run_dqn --config-dir configs/dqn_1v1

# Generic launcher — reads configs/experiment.yaml, whatever algorithm.name it specifies (default: iql)
PYTHONPATH=src python -m multi_agent_package.scripts.run_from_config
```

---

## 3. Train

Each `run_<algo>.py` script trains and saves a checkpoint by default:

```bash
# IQL, 1000 episodes (override via experiment_iql.yaml, not a CLI flag)
PYTHONPATH=src python -m multi_agent_package.scripts.run_iql --save-path my_iql.pkl

# CQL
PYTHONPATH=src python -m multi_agent_package.scripts.run_cql --save-path my_cql.pkl

# MixedTrainer (predator/prey algorithm assignment comes from experiment_mixed.yaml)
PYTHONPATH=src python -m multi_agent_package.scripts.run_mixed --save-path my_mixed.pkl

# DQN
PYTHONPATH=src python -m multi_agent_package.scripts.run_dqn --save-path my_dqn.pkl
```

Each algorithm also has its own standalone CLI with hyperparameters as flags (e.g. `python -m baselines.IQL.iql --episodes 1000 --alpha 0.1 ...`), which builds its own `GridWorldEnv` directly rather than going through `run_from_config` — see [reference/api-reference.md](../reference/api-reference.md).

Training logs to stdout every 100 episodes (10 for DQN, via `log_interval`).

---

## 4. Evaluate

```bash
# Evaluate a saved checkpoint (headless)
python -m baselines.IQL.iql --mode eval --load-path my_iql.pkl --episodes 20

# Evaluate with pygame visualization (requires a display)
python -m baselines.IQL.iql --mode eval --load-path my_iql.pkl --episodes 5 --render
```

Or use `evaluate.py`, which builds its own env + algorithm from a config directory (it does **not** take an existing `algo`/`env` — see [guides/using-evaluate.md](using-evaluate.md) for the exact signature and output shape):

```python
from multi_agent_package.scripts.evaluate import evaluate
results = evaluate(config_dir="configs", episodes=20)
print(results)
# {"mean_episode_length": 47.2, "std_episode_length": 8.1, "mean_return_pred_1": -12.4, ...}
```

---

## 5. Modify and iterate

Common experiment variations — all achievable via YAML, no code changes:

**Change observation type:**
```yaml
# observations.yaml
observations:
  type: local_only   # blind: agents see only their own position
```

**Add distance shaping for predators:**
```yaml
# rewards.yaml
shaping:
  - name: predator_distance
    weight: 1.0
```

**Make episodes longer:**
```yaml
# env.yaml
termination:
  max_steps: 1000
```

**Switch algorithm:**
```yaml
# experiment.yaml
experiment:
  algorithm:
    name: cql
    params:
      episodes: 2000
```

---

## Common failure modes

| Symptom | Cause |
|---------|-------|
| `KeyError: 'local_raidus'` | Typo in `observations.yaml` type |
| `ValueError: Algorithm 'X' not registered` | `import baselines` missing before registry lookup |
| `KeyError: 'experiment'` | Config nested as `configs["experiment"]["experiment"]["algorithm"]` — two levels |
| Training finishes instantly (0 steps) | `max_steps: 0` or `capture_threshold: 0` in `env.yaml` |
| Q-tables always empty | `episodes` too low or env always truncating before any step |
| `pip install -e .` doesn't make `multi_agent_package` importable | No build backend is configured yet — use `PYTHONPATH=src` instead (see [docs/git-workflow.md](../git-workflow.md#setup)) |
