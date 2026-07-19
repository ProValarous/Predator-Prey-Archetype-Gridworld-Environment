# Quickstart: Checkout to Trained Model

Five steps from a fresh clone to a trained agent.

---

## 1. Install dependencies

```bash
cd src
pip install -e .
pip install -r ../requirements.txt
```

Verify the install:

```bash
python -c "from multi_agent_package.core.gridworld import GridWorldEnv; print('OK')"
```

---

## 2. Configure your experiment

Five YAML files in `configs/` control everything. For a first run, the defaults work:

| File | Controls |
|------|---------|
| `env.yaml` | Grid size, obstacles, seed, termination rules |
| `agents.yaml` | How many predators and prey, their speed/stamina |
| `observations.yaml` | What each agent can see |
| `rewards.yaml` | Reward signals and shaping |
| `experiment.yaml` | Which algorithm, its hyperparameters |

To run IQL with default settings, no edits needed. To run CQL or MixedTrainer, point at the matching experiment file:

```bash
# IQL (default)
python -m multi_agent_package.scripts.run_from_config

# CQL
python -m multi_agent_package.scripts.run_iql   # or run_cql / run_mixed
```

---

## 3. Train

```bash
cd src

# Train with IQL for 1000 episodes, save checkpoint
python -m baselines.IQL.iql --mode train --episodes 1000 --save-path my_iql.pkl

# Train with CQL
python -m baselines.CQL.cql --mode train --episodes 1000 --save-path my_cql.pkl

# Train MixedTrainer (predators CQL, prey IQL)
python -m baselines.MIXED.mix_train --mode train --predator-algo cql --prey-algo iql \
    --episodes 1000 --save-path my_mixed.pkl
```

Training logs to stdout every 100 episodes (configurable via `logging`).

---

## 4. Evaluate

```bash
# Evaluate a saved checkpoint (headless)
python -m baselines.IQL.iql --mode eval --load-path my_iql.pkl --episodes 20

# Evaluate with pygame visualization (requires a display)
python -m baselines.IQL.iql --mode eval --load-path my_iql.pkl --episodes 5 --render
```

Or use `evaluate.py` for structured metrics:

```python
# From Python
from multi_agent_package.scripts.evaluate import evaluate
results = evaluate(algo, env, episodes=20)
print(results)
# {"mean_return": {...}, "episode_lengths": [...], "capture_rate": 0.7}
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

See [troubleshooting.md](../troubleshooting.md) for more.
