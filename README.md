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

  (Work-In-Progress) A minimalist, discrete multi-agent predator-prey archytype environment.

<h3> Early Environment SnapShot</h3>
<img src="miscellenous/imgs/game_snap_v2.png" alt="Early Environment SnapShot" width="400"/>

---

## Overview

This repository provides a **discrete, grid-based predator-prey simulation environment** designed to support controlled, interpretable, and reproducible experiments in MARL. The environment models classic predator-prey dynamics where multiple agents (predators and prey) interact and learn in a bounded grid world.

It is research-oriented, multi-agent GridWorld environment designed for undergraduate-accessible Multi-Agent Reinforcement Learning (MARL) experimentation with strong guarantees on reproducibility, extensibility, and scientific rigor.

This repository provides a controlled synthetic laboratory for studying coordination, pursuit–evasion, and emergent behavior under constraints such as speed, stamina, and partial observability—without requiring students to modify environment internals.

Key goals of this framework:

* Facilitate **mechanistic understanding** of MARL behavior
* Support **reproducible research** and **ablation studies**
* Provide an **accessible learning tool** for students and new researchers

---
📁 Repository Structure
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
│       │   ├── base.py              # Observation interface / contract
│       │   ├── default.py           # Full-information observation
│       │   ├── local_only.py        # Self-only observation
│       │   ├── local_radius.py      # Partial observability example
│       │   └── README.md
│       │
│       ├── rewards/                 # 🎯 REWARD PLUG-INS (STUDENTS)
│       │   ├── base.py              # Reward interface / contract
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
├── CONTRIBUTING.md                  # 🚨 CONTRIBUTOR RULES
├── README.md
└── pyproject.toml / setup.cfg
---
## Features

### Fully Interpretable

* **State and action spaces are fully enumerable and transparent**, making it easy to track agent behavior, transitions, and environment evolution step-by-step.

### Modular and Customizable

* **Variables and reward structures can be modified in isolation**, enabling controlled experimentation.
* Easily change grid size, agent behavior, episode length, reward schemes, terminal conditions, etc.

### Built for Rigorous Experimentation

* **Reproducible ablation experiments** are supported by design.
* Codebase encourages transparency, logging, and clear interpretation of learning dynamics.

### Education-Ready

* **Simplified structure** and clean API make it beginner-friendly.
* Excellent for undergraduates or early-stage researchers to explore MARL concepts without the overhead of complex simulator frameworks.

---

## Use Cases

* Studying **emergent cooperation and competition**
* Benchmarking MARL algorithms in a clean, discrete setting
* Analyzing **credit assignment**, **multi-agent exploration**, and **policy coordination**
* Teaching reinforcement learning and agent-based modeling

---

## Environment Dynamics

* Agents: Can be initialized as `predator`, `prey`, or any custom role
* Gridworld: 2D discrete space with obstacle-free navigation
* Actions: {UP, DOWN, LEFT, RIGHT, STAY}
* Rewards: Configurable, including custom interaction-based incentives
* Terminal conditions: Catching prey, timeouts, or customizable rules

---

## Getting Started

⚙️ Installation
1️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows

2️⃣ Install the package (editable mode)
pip install -e .


This registers multi_agent_package correctly using the src/ layout.

▶️ Running an Experiment

All experiments must be launched from the repository root.

python -m multi_agent_package.scripts.run_from_config


This command:

Loads YAML files from configs/

Constructs agents and environment

Registers observation and reward plug-ins

Resets the environment

Executes a rollout (optionally rendered)

🖥️ Rendering

Rendering follows Gymnasium conventions and is explicitly controlled.

Enable rendering in configs/env.yaml:

env:
  render_mode: human


Rendering is:

deterministic,

safe for headless runs,

and isolated from environment logic.

🧪 Reproducibility

An experiment is fully determined by:

YAML configuration files

explicit random seeds

registered observation and reward logic

If two runs use the same configs, they must produce identical results.

Any contribution that violates this assumption will be rejected.

👩‍🎓 For Undergraduate Contributors

You are expected to:

implement new reward functions,

design observation schemes,

run controlled experiments,

perform ablations,

and report clear metrics.

You are not expected to:

modify environment internals,

touch transition logic,

or debug rendering pipelines.

This mirrors how real research codebases operate.

---

## Contributing

We welcome contributions that improve the clarity, utility, or extensibility of the environment. If you have ideas for enhancements, fixes, or new features, please open an issue or submit a pull request.

---

## Citation

If you use this environment in your research, teaching, or project, please cite it using the following BibTeX:

```bibtex
@misc{predatorpreygridworld,
  author       = {Ahmed Atif and others},
  title        = {Predator-Prey Gridworld Environment},
  year         = {2025},
  howpublished = {\url{https://github.com/ProValarous/Predator-Prey-Gridworld-Environment}},
  note         = {A discrete testbed for studying Multi-Agent Reinforcement Learning dynamics.}
}
```

---

## License

This project is licensed under the Apache-2.0 license.

---

## Contact

For any questions, issues, or collaborations, please reach out via the [GitHub repository](https://github.com/ProValarous/Predator-Prey-Gridworld-Environment/issues).

---

## Acknowledgements

This project draws inspiration from classic RL environments and aims to provide a more transparent, MARL-specific framework. Contributions from the community are deeply appreciated!
