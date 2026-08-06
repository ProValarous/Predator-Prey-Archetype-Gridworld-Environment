# Plug and Play

This folder is the front door of the repository: everything runnable lives
here, next to the configs it consumes. The library itself (`src/ppage`) stays
untouched; you plug experiments together from here.

It is also one half of PPAGE's design philosophy — **accessible by default,
versatile by design**. This folder is the accessibility half: run and vary
experiments through YAML alone. The versatility half is the plugin surface
(custom observations, rewards, action spaces, algorithms); see
[Design Philosophy](https://uhumalab.github.io/PPAGE/overview/design-philosophy/)
for what you can build.

- **`configs/`** - YAML experiment definitions. `env.yaml`, `agents.yaml`,
  `observations.yaml`, `rewards.yaml`, `actions.yaml`, and one
  `experiment_*.yaml` per algorithm, plus ready-made experiment sets like
  `dqn_1v1/`.
- **`scripts/`** - the runners that consume those configs.

## Setup (once)

From the repository root:

```bash
pip install -e ".[baselines]"   # baselines extra adds PyTorch (DQN, AC, A2C, A3C)
```

## Run

All commands run from the repository root.

```bash
# The generic entrypoint: algorithm and all parameters chosen via YAML
python plug-and-play/scripts/run_from_config.py

# Ready-made DQN experiment set
python plug-and-play/scripts/run_dqn.py --config-dir plug-and-play/configs/dqn_1v1

# Watch one episode in a pygame window
python plug-and-play/scripts/render.py

# Evaluate a saved checkpoint
python plug-and-play/scripts/evaluate.py --load-path trained_iql.pkl
```

Every script accepts `--help` for its full set of flags. To change what an
experiment does, edit the YAML in `configs/` rather than the scripts; identical
configuration plus identical seed reproduces the identical trajectory.
