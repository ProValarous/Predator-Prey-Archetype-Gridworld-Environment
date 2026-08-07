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
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -e ".[baselines]"   # baselines extra adds PyTorch (DQN, AC, A2C, A3C)
```

## 3. Run Training

All experiments are driven by the YAML files in `plug-and-play/configs/`.

```bash
python plug-and-play/scripts/run_from_config.py
```

This runs IQL for 500 episodes on a 10×10 grid with 3 predators and 3 prey (from `plug-and-play/configs/experiment.yaml`).  
Progress is logged every 100 episodes.

## 4. Render an Episode

No YAML edit needed; the script forces a window:

```bash
python plug-and-play/scripts/render.py
```

A pygame window opens and plays one episode with random actions.  
Pass `--load-path <ckpt>` to watch a trained policy instead.

## 5. Train via CLI (standalone)

For quick experiments without editing YAML files:

```bash
# IQL
python -m ppage.baselines.IQL.iql --episodes 1000 --size 8 --predators 1 --preys 1

# CQL
python -m ppage.baselines.CQL.cql --episodes 1000 --alpha 0.1

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
    actions/                  # Pluggable action handlers
    observations/             # Pluggable observation builders
    rewards/                  # Pluggable reward functions
    wrappers/                 # Environment wrappers
    registry/                 # Component registries
    baselines/
      IQL/                    # Independent Q-Learning
      CQL/                    # Centralized Q-Learning
      MIXED/                  # Mixed predator/prey algorithms
      DQN/                    # Deep Q-Network
      AC/                     # Actor-Critic
      A2C/                    # Advantage Actor-Critic
      A3C/                    # Asynchronous Advantage Actor-Critic
plug-and-play/                # Start here: runnable entry points
  scripts/                    # run_from_config.py, render.py, evaluate.py ...
  configs/                    # All experiment YAML files
```

---

## Troubleshooting

**`ModuleNotFoundError: ppage`** — install the package first: `pip install -e .` from the repository root.

**an unwanted pygame window opens during training** — set `render_mode: null` in `plug-and-play/configs/env.yaml` (already the shipped default). Note that `render.py` overrides this key to `"human"`, so it cannot suppress the window there.

**pygame window never appears when running `render.py`** — the script forces a window, so a missing one means pygame failed to initialise; check the traceback.

**`pip` not found** — use `python -m pip install` instead of calling `pip` directly.
