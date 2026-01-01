# 🐾 Predator–Prey Gridworld Environment

<p align="center">
  <a href="https://provalarous.github.io/Predator-Prey-Archetype-Gridworld-Environment/">
    <img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Documentation">
  </a>
  <a href="./CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions Welcome">
  </a>
  <a href="./CODE_OF_CONDUCT.md">
    <img src="https://img.shields.io/badge/code%20of%20conduct-enforced-orange.svg" alt="Code of Conduct">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

A **discrete, grid-based multi-agent predator–prey environment** designed as a controlled research testbed for studying coordination, pursuit–evasion, and emergent behavior in Multi-Agent Reinforcement Learning (MARL).

<h3>Current Rendering Prototype</h3>
<img src="miscellenous/imgs/game_snap_v2.png" alt="Gridworld Rendering Snapshot" width="400"/>

---

## Overview

This repository provides a **research-oriented, interpretable GridWorld environment** for multi-agent predator–prey dynamics. It is explicitly designed to support **controlled experimentation**, **reproducibility**, and **undergraduate-accessible research workflows**.

The environment serves as a **synthetic laboratory** for studying:
- coordination and competition
- pursuit–evasion dynamics
- constraint-induced coupling (speed, stamina)
- partial observability and information asymmetry

Crucially, students and contributors **do not modify environment internals**. Instead, they extend behavior through **well-defined plug-in interfaces** for rewards and observations.

### Design Goals

- Enable **mechanistic understanding** of MARL algorithms
- Support **reproducible experiments and ablation studies**
- Provide an **educationally safe research codebase** for undergraduates
- Enforce clean separation between environment dynamics and learning logic

---

## Repository Structure

```text
Predator-Prey-Archetype-Gridworld-Environment/
├── src/
│   └── multi_agent_package/
│
│       ├── core/                   # 🔒 IMMUTABLE CORE (DO NOT EDIT)
│       │   ├── gridworld.py         # Grid dynamics, transitions, rendering
│       │   ├── agent.py             # Base agent definition
│       │   └── __init__.py
│       │
│       ├── observations/            # 👁️ OBSERVATION PLUG-INS (STUDENTS)
│       │   ├── base.py              # Observation contract
│       │   ├── default.py           # Full-information observation
│       │   ├── local_only.py        # Self-only observation
│       │   ├── local_radius.py      # Partial observability example
│       │   └── README.md
│       │
│       ├── rewards/                 # 🎯 REWARD PLUG-INS (STUDENTS)
│       │   ├── base.py              # Reward contract
│       │   ├── base_reward.py       # Canonical capture-based reward
│       │   ├── predator_distance.py # Distance-based shaping example
│       │   ├── survival_reward.py   # Survival-based reward example
│       │   └── README.md
│       │
│       ├── registry/                # 🔌 SAFE PLUG-IN REGISTRATION
│       │   ├── reward_registry.py
│       │   ├── observation_registry.py
│       │   └── __init__.py
│       │
│       ├── scripts/                 # ▶️ HIGH-LEVEL ENTRY POINTS
│       │   ├── run_from_config.py   # Main experiment launcher
│       │   ├── render.py            # Visualization-only script
│       │   ├── evaluate.py          # Metrics & plots
│       │   └── sweep.py             # Parameter sweeps
│       │
│       └── __init__.py
│
├── configs/                         # 🎛️ EXPERIMENT DEFINITIONS (YAML ONLY)
│   ├── env.yaml                     # Environment parameters
│   ├── agents.yaml                  # Agent counts & attributes
│   ├── rewards.yaml                 # Reward selection
│   ├── observations.yaml            # Observation selection
│   └── experiment.yaml              # Experiment glue
│
├── CONTRIBUTING.md                  # 🚨 Contributor rules
├── CODE_OF_CONDUCT.md
├── LICENSE
├── README.md
└── pyproject.toml / setup.cfg
````

---

## Features

### Fully Interpretable

All environment state, transitions, and rewards are **explicitly represented and enumerable**. There are no hidden simulators or opaque physics engines, making learning dynamics easy to inspect and debug.

### Modular by Construction

Reward functions and observation schemes are **plug-ins**, not hard-coded logic. This allows controlled experimentation without modifying core environment behavior.

### Reproducibility First

An experiment is fully determined by:

* YAML configuration files
* explicit random seeds
* registered observation and reward implementations

Runs using identical configurations must produce identical outcomes. Contributions violating this assumption will be rejected.

### Education-Ready

The codebase enforces **clean boundaries** between infrastructure and experimentation, mirroring how real research codebases operate while remaining accessible to undergraduate contributors.

---

## Use Cases

* Studying emergent cooperation and competition
* Benchmarking MARL algorithms in a clean, discrete setting
* Analyzing credit assignment and coordination failures
* Teaching reinforcement learning and agent-based modeling

---

## Environment Dynamics

* **Agents**: predator, prey, or custom roles
* **World**: discrete 2D grid with obstacles
* **Actions**: up, down, left, right, stay
* **Rewards**: fully configurable via plug-ins
* **Termination**: capture events, timeouts, or user-defined conditions

---

## Getting Started

### Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows
```

Install in editable mode:

```bash
pip install -e .
```

This correctly registers `multi_agent_package` using the `src/` layout.

---

### Running an Experiment

All experiments must be launched **from the repository root**:

```bash
python -m multi_agent_package.scripts.run_from_config
```

This command:

* loads YAML configuration files from `configs/`
* constructs agents and environment
* registers reward and observation plug-ins
* runs a rollout (optionally rendered)

> Note: Running this command from `src/` or any subdirectory will result in import or missing-config errors. This is expected behavior due to the `src/` layout.

---

### Rendering

Rendering follows Gymnasium conventions and is explicitly controlled.

Enable rendering in `configs/env.yaml`:

```yaml
env:
  render_mode: human
```

Rendering is deterministic, safe for headless execution, and isolated from environment logic.

---

## For Undergraduate Contributors

You are encouraged to:

* implement new reward functions
* design observation schemes
* reproduce experiments with modified parameters
* perform ablations and report metrics

You are **not expected** to:

* modify environment internals
* change transition dynamics
* debug rendering or seeding logic

This separation reflects how real MARL research infrastructure is structured.

---

## Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) before submitting issues or pull requests. Contributions should prioritize clarity, reproducibility, and pedagogical value.

---

## Citation

If you use this environment in research or teaching, please cite:

```bibtex
@misc{predatorpreygridworld,
  author       = {Ahmed Atif and contributors},
  title        = {Predator–Prey Gridworld Environment},
  year         = {2025},
  howpublished = {\url{https://github.com/ProValarous/Predator-Prey-Archetype-Gridworld-Environment}},
  note         = {A discrete testbed for studying Multi-Agent Reinforcement Learning dynamics}
}
```

---

## License

This project is licensed under the **Apache License 2.0**.

---

## Contact

Questions, issues, and discussions should be raised via the GitHub issue tracker.

---

## Acknowledgements

This project is inspired by classic reinforcement learning environments and is shaped by the goal of making MARL research accessible, interpretable, and reproducible for students and researchers alike.

```

