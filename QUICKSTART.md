# Quickstart

Get from zero to a running training run in five minutes.

---

## 1. Prerequisites

- Python 3.10+
- A virtual environment

```bash
python -m venv .venv
```

## 2. Install Dependencies

Everything below runs from the repository root.

```bash
# Windows
.venv\Scriptsctivate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[baselines]"   # baselines extra adds PyTorch (DQN, AC, A2C, A3C)
```

## 3. Run Training

All experiments are driven by the YAML files in `plug-and-play/configs/`.

```bash
python plug-and-play/scripts/run_from_config.py
```

This runs IQL for 500 episodes on a 10×10 grid with 3 predators and 3 prey.  
Progress is logged every 100 episodes.

## 4. Render an Episode

Set `render_mode: human` in `plug-and-play/configs/env.yaml` (it is the default), then:

```bash
python plug-and-play/scripts/render.py
```

A pygame window opens and plays one training run visually.  
Set `render_mode: null` to disable the window and run headless.

## 5. Train via CLI (standalone)

For quick experiments without editing YAML files:

```bash
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
| `plug-and-play/configs/env.yaml` | Grid size, obstacles, render mode, episode cap |
| `plug-and-play/configs/agents.yaml` | Predator / prey counts, speed, stamina |
| `plug-and-play/configs/observations.yaml` | Observation type and radius per agent type |
| `plug-and-play/configs/rewards.yaml` | Base reward + shaping weights |
| `plug-and-play/configs/experiment.yaml` | Algorithm name and hyperparameters |

---

## Project Layout

```
src/
  ppage/                      # Environment core (the installable library)
    core/                     # GridWorldEnv, Agent
    observations/             # Pluggable observation builders
    rewards/                  # Pluggable reward functions
    baselines/
      IQL/                    # Independent Q-Learning
      CQL/                    # Centralized Q-Learning
plug-and-play/                # Start here: runnable entry points
  scripts/                    # run_from_config.py, render.py, evaluate.py ...
  configs/                    # All experiment YAML files
```

---

## Troubleshooting

**`ModuleNotFoundError: ppage`** — install the package first: `pip install -e .` from the repository root.

**pygame window does not open** — set `render_mode: null` in `env.yaml` for headless runs.

**`pip` not found** — use `python -m pip install` instead of calling `pip` directly.
