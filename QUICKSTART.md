# Quickstart

Get from zero to a running training run in five minutes.

---

## 1. Prerequisites

- Python 3.10+
- A virtual environment (example uses `.new_venv`)

```bash
python -m venv .new_venv
```

## 2. Install Dependencies

```bash
# Windows
.new_venv\Scripts\python.exe -m pip install -r requirements.txt

# macOS / Linux
.new_venv/bin/pip install -r requirements.txt
```

## 3. Run Training

All experiments are driven by the YAML files in `configs/`.

```bash
cd src

# Windows
..\.new_venv\Scripts\python.exe -m ppage.scripts.run_from_config

# macOS / Linux
../.new_venv/bin/python -m ppage.scripts.run_from_config
```

This runs IQL for 500 episodes on a 10×10 grid with 3 predators and 3 prey.  
Progress is logged every 100 episodes.

## 4. Render an Episode

Set `render_mode: human` in `configs/env.yaml` (it is the default), then:

```bash
cd src
python -m ppage.scripts.render
```

A pygame window opens and plays one training run visually.  
Set `render_mode: null` to disable the window and run headless.

## 5. Train via CLI (standalone)

For quick experiments without editing YAML files:

```bash
cd src

# IQL
python -m ppage.baselines.IQL.iql --episodes 1000 --size 8 --predators 1 --preys 1

# CQL
python -m ppage.baselines.CQL.cql --episodes 1000 --cql-alpha 0.1

# Mixed (predators CQL, prey IQL)
python -m ppage.baselines.MIXED.mix_train --predator-algo cql --prey-algo iql --episodes 1000
```

All scripts save trained Q-tables to a `.pkl` file (see `--save-path`).

## 6. Evaluate a Saved Model

```bash
cd src

# IQL
python -m ppage.baselines.IQL.iql --mode eval --load-path trained_iql.pkl

# CQL
python -m ppage.baselines.CQL.cql --mode eval --load-path trained_cql.pkl

# Mixed
python -m ppage.baselines.MIXED.mix_train --mode eval --load-path trained_mixed.pkl
```

---

## Key Config Files

| File | Controls |
|------|----------|
| `configs/env.yaml` | Grid size, obstacles, render mode, episode cap |
| `configs/agents.yaml` | Predator / prey counts, speed, stamina |
| `configs/observations.yaml` | Observation type and radius per agent type |
| `configs/rewards.yaml` | Base reward + shaping weights |
| `configs/experiment.yaml` | Algorithm name and hyperparameters |

---

## Project Layout

```
src/
  ppage/                      # Environment core
    core/                     # GridWorldEnv, Agent
    observations/             # Pluggable observation builders
    rewards/                  # Pluggable reward functions
    scripts/                  # run_from_config.py, render.py
    baselines/
      IQL/                    # Independent Q-Learning
      CQL/                    # Centralized Q-Learning
configs/                      # All experiment YAML files
wiki/                         # Architecture specs and ADRs
```

---

## Troubleshooting

**`ModuleNotFoundError`** — run scripts from inside `src/`, or add `src/` to `PYTHONPATH`.

**pygame window does not open** — set `render_mode: null` in `env.yaml` for headless runs.

**`pip` not found** — use `python -m pip install` instead of calling `pip` directly.
