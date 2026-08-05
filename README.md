# 🐾 Predator–Prey Gridworld Environment

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python versions">
  <a href="https://uhumalab.github.io/PPAGE/"><img src="https://img.shields.io/badge/docs-mkdocs-teal.svg" alt="Docs"></a>
  <a href="https://github.com/UHUMALAB/PPAGE/actions/workflows/ci.yaml"><img src="https://github.com/UHUMALAB/PPAGE/actions/workflows/ci.yaml/badge.svg?branch=STRP" alt="CI"></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen" alt="pre-commit"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
</p>

<p align="center">
  <img src="miscellenous/imgs/demo.gif" alt="A predator (red) chasing down a prey (green) across a 10x10 obstacle grid until capture" width="480">
</p>

<p align="center"><sub>A speed-2 predator pursuing a speed-1 prey around obstacles (<code>configs/dqn_1v1</code>) until capture.</sub></p>

<p align="center">
A <b>deterministic, modular, research-grade multi-agent predator–prey environment</b> for studying coordination, pursuit–evasion, and emergent behavior in Multi-Agent Reinforcement Learning. It is not just a simulation, it is a controlled experimental laboratory for understanding how multi-agent learning systems behave, with fully inspectable dynamics, pluggable perception and incentives, and reproducibility enforced by construction rather than assumed.
</p>

---

## 🎯 What This Repository Is About

This project provides:

* A discrete 2D gridworld with predators and prey
* Explicit, fully inspectable transition dynamics
* Pluggable observation models
* Pluggable reward functions
* Pluggable action spaces (including per-agent speed/stamina mechanics)
* Strict separation between environment and learning
* Deterministic, reproducible experiments

It is designed to make MARL **mechanistically understandable**, not opaque.

---

## ❓ Why This Exists

Most MARL environments:

* Mix environment logic and learning code
* Are difficult to modify safely
* Hide important transition mechanics
* Make reproducibility fragile
* Encourage experimentation by hacking internals

This project exists to enforce something stricter:

```text
Environment dynamics → Perception → Incentives → Learning
```

Each layer is isolated by construction.

* **Environment dynamics** defines what can happen
* **Perception** defines what agents know
* **Incentives** define what agents optimize
* **Learning** defines how they adapt

By separating these layers, we can study each one independently.

That separation is the core idea of this repository.

---

## 🧠 What It Tries to Achieve

This environment aims to:

* Enable controlled MARL experimentation
* Support clean ablation studies
* Enforce reproducibility by design
* Prevent accidental coupling between components
* Provide a safe research codebase for students
* Make emergent behavior inspectable and analyzable

The goal is not realism.

The goal is **clarity, modularity, and scientific control**.

---

## 🏗 Architectural Philosophy

The repository is divided into two major components:

### 1️⃣ `ppage`: The Environment

Implements:

* Grid environment dynamics, agent movement, capture logic, episode termination (`core/`, immutable)
* Observation plug-ins: perception (`observations/`)
* Reward plug-ins: incentives (`rewards/`)
* Action-space plug-ins: what an agent's action integers mean (`actions/`)
* Wrappers: cross-cutting mechanics layered on top of the base env, e.g. per-agent speed/stamina (`wrappers/`)
* Registries: the only sanctioned way to wire a plug-in into an experiment (`registry/`)

This layer defines the world.

Currently registered plug-ins:

| Category     | Registered options                                                    |
| ------------ | ----------------------------------------------------------------------- |
| Observations | `default`, `local_only`, `local_radius`, `absolute`, `relative`          |
| Rewards      | `base`, `predator_distance`, `survival`                                  |
| Actions      | `discrete_5`, `cross`, `speed_discrete_5`                               |
| Wrappers     | `SpeedWrapper` (per-agent speed/stamina, applied last in the build chain) |

### 2️⃣ `ppage.baselines`: The Learning Algorithms

Implements:

* **IQL**: Independent Q-Learning (tabular)
* **CQL**: Centralized Q-Learning (tabular)
* **MixedTrainer**: per-team algorithm assignment (e.g. CQL predators vs IQL prey)
* **DQN**: Deep Q-Network (PyTorch, generic observation encoder, replay buffer)

See [`src/ppage/baselines/README.md`](src/ppage/baselines/README.md) for the algorithm contract and when to use each one.

Algorithms interact with the environment only through:

```python
env.reset()
env.step(actions)
```

They never access internal state directly.

This guarantees structural integrity.

---

## 🔁 Reproducibility as a First-Class Constraint

An experiment is fully determined by:

* YAML configuration files
* Explicit random seeds
* Registered observation modules
* Registered reward modules

Identical configuration → identical trajectories.

This is enforced, not assumed.

---

## 📂 Repository Structure

```
src/
└── ppage/                    # Environment
    ├── core/                 # Immutable environment dynamics (maintainers only)
    ├── observations/         # Perception plug-ins
    ├── rewards/              # Incentive plug-ins
    ├── actions/              # Action-space plug-ins
    ├── wrappers/             # Cross-cutting mechanics (e.g. SpeedWrapper)
    ├── registry/             # Safe plug-in selection
    ├── scripts/              # Experiment runners (run_from_config, run_dqn, ...)
    └── baselines/            # Learning algorithms
        ├── IQL/  CQL/  MIXED/  DQN/
        └── registry/         # Algorithm name -> class

configs/                      # YAML experiment definitions
├── env.yaml, agents.yaml, observations.yaml, rewards.yaml, actions.yaml
├── experiment_{iql,cql,mixed,dqn}.yaml
└── dqn_1v1/, dqn_speed1/, dqn_speed2/, dqn_speed3/   # ready-made DQN experiment sets

tests/                        # pytest suite: registries, plugin contracts,
                               # end-to-end training, architecture rules
```

Core environment dynamics is stable infrastructure.

Observations, rewards, and actions are the intended extension points.

---

## 🧪 What You Can Study With This

* Emergent cooperation between predators
* Coordination failures
* Reward shaping effects
* Partial observability impact
* Centralized vs decentralized learning
* Constraint-induced coupling (speed, stamina)
* Credit assignment challenges

This environment is meant for:

* MARL research
* Undergraduate research labs
* Algorithm benchmarking
* Teaching reinforcement learning
* Controlled ablation experiments

---

## ⚡ Quickstart

The package uses a standard `src/` layout with a `pyproject.toml` build backend,
so an editable install makes `ppage` (including `ppage.baselines`) importable
without setting `PYTHONPATH`.

```bash
git clone https://github.com/UHUMALAB/PPAGE.git
cd PPAGE

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -e ".[baselines]"   # plain `pip install -e .` skips the PyTorch baselines

# Run the default experiment (3 predators vs 3 prey, IQL, configs/experiment.yaml)
python -m ppage.scripts.run_from_config

# Or one of the ready-made DQN experiments
python -m ppage.scripts.run_dqn --config-dir configs/dqn_1v1
```

All experiments are launched from the repository root.

### Running the tests

```bash
pip install -e ".[dev,baselines]"
python -m pytest tests/ -q
```

CI (`.github/workflows/ci.yaml`) runs this same suite plus Black/flake8/pylint on every push and PR to `main`/`STRP`, and blocks any PR that touches `core/` (see below).

---

## 👩‍🎓 For Contributors and Students

You are encouraged to:

* Implement new reward functions
* Design new observation schemes
* Design new action spaces or wrappers
* Run structured experiments
* Perform reproducible ablations

You are not expected to modify core environment dynamics; this is enforced automatically: a CI check fails any pull request that touches `src/ppage/core/`. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contribution rules, and [`docs/git-workflow.md`](docs/git-workflow.md) for branching, commits, and how to open a PR.

This mirrors how research infrastructure is structured in practice.

---

## 📜 Citation

```bibtex
@misc{predatorpreygridworld,
  author       = {Muhammad Ahmed Atif and Nehal Naeem Haji and Muhammad Affan and Areesha Kashif and Musab Kasbati and Afshad Yazdi Sidhwa},
  title        = {Predator–Prey Gridworld Environment},
  year         = {2025},
  note         = {A deterministic modular testbed for Multi-Agent Reinforcement Learning}
}
```

---

## 📜 License

Apache License 2.0
