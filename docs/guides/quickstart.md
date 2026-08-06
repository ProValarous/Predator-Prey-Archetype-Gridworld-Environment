# Quickstart: Checkout to Trained Model

Five steps from a fresh clone to a trained agent.

---

## 1. Install

Prerequisite: Python 3.10-3.12.

The package uses a standard `src/` layout with a `pyproject.toml` build backend,
so an editable install makes `ppage` (including `ppage.baselines`) importable
without setting `PYTHONPATH`. Run commands from the repository root.

```bash
pip install -e ".[baselines]"   # add dev to the extras for the test/lint tools
```

Verify the install:

```bash
python -c "from ppage.core.gridworld import GridWorldEnv; print('OK')"
```

---

## 2. Configure your experiment

Six YAML files in `plug-and-play/configs/` control everything. For a first run, the defaults work:

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
# IQL — reads plug-and-play/configs/experiment_iql.yaml
python plug-and-play/scripts/run_iql.py

# CQL — reads plug-and-play/configs/experiment_cql.yaml
python plug-and-play/scripts/run_cql.py

# MixedTrainer — reads plug-and-play/configs/experiment_mixed.yaml
python plug-and-play/scripts/run_mixed.py

# DQN — reads plug-and-play/configs/experiment_dqn.yaml
python plug-and-play/scripts/run_dqn.py

# Actor-Critic — reads plug-and-play/configs/experiment_actor_critic.yaml
python plug-and-play/scripts/run_actor_critic.py

# A2C — reads plug-and-play/configs/experiment_a2c.yaml
python plug-and-play/scripts/run_a2c.py

# A3C — reads plug-and-play/configs/experiment_a3c.yaml
python plug-and-play/scripts/run_a3c.py

# Or a ready-made DQN experiment set (1 predator vs 1 prey, double+dueling enabled)
python plug-and-play/scripts/run_dqn.py --config-dir plug-and-play/configs/dqn_1v1

# Generic launcher — reads plug-and-play/configs/experiment.yaml, whatever algorithm.name it specifies (default: iql)
python plug-and-play/scripts/run_from_config.py
```

---

## 3. Train

Each `run_<algo>.py` script trains and saves a checkpoint by default:

```bash
# IQL, 1000 episodes from experiment_iql.yaml (override via that file, not a CLI flag)
python plug-and-play/scripts/run_iql.py --save-path my_iql.pkl

# CQL
python plug-and-play/scripts/run_cql.py --save-path my_cql.pkl

# MixedTrainer (predator/prey algorithm assignment comes from experiment_mixed.yaml)
python plug-and-play/scripts/run_mixed.py --save-path my_mixed.pkl

# DQN
python plug-and-play/scripts/run_dqn.py --save-path my_dqn.pkl
```

Each algorithm also has its own standalone CLI with hyperparameters as flags (e.g. `python -m ppage.baselines.IQL.iql --episodes 1000 --alpha 0.1 ...`), which builds its own `GridWorldEnv` directly rather than going through `run_from_config` — see [reference/api-reference.md](../reference/api-reference.md).

Training logs to stdout every 100 episodes (10 for DQN, via `log_interval`).

---

## 4. Evaluate

```bash
# Evaluate a saved checkpoint (headless)
python -m ppage.baselines.IQL.iql --mode eval --load-path my_iql.pkl --episodes 20

# Evaluate with pygame visualization (requires a display)
python -m ppage.baselines.IQL.iql --mode eval --load-path my_iql.pkl --episodes 5 --render
```

Or use `evaluate.py`, which builds its own env + algorithm from a config directory (it does **not** take an existing `algo`/`env` — see [guides/using-evaluate.md](using-evaluate.md) for the exact signature and output shape):

```python
import sys
sys.path.insert(0, "plug-and-play/scripts")  # run from the repository root

from evaluate import evaluate
results = evaluate(config_dir="plug-and-play/configs", episodes=20, load_path="my_iql.pkl")
print(results)
# {"mean_episode_length": 47.2, "std_episode_length": 8.1, "mean_return_pred_1": -12.4, ...}
```

Omitting `load_path` evaluates a fresh (untrained) policy.

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
| `ValueError: Algorithm 'X' not registered` | `import ppage.baselines` missing before registry lookup |
| `KeyError: 'experiment'` | Config nested as `configs["experiment"]["experiment"]["algorithm"]` — two levels |
| Training finishes instantly (0 steps) | `max_steps: 0` or `capture_threshold: 0` in `env.yaml` |
| Q-tables always empty | `episodes` too low or env always truncating before any step |
| `ModuleNotFoundError: ppage` | The package isn't installed — run `pip install -e .` from the repository root |
