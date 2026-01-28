# 🐾 Predator–Prey Gridworld Environment


[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](contributing.md)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-enforced-orange.svg)](CODE_OF_CONDUCT.md)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ProValarous/Predator-Prey-Gridworld-Environment/blob/main/LICENSE)

A **discrete, grid-based multi-agent predator–prey environment** designed as a controlled research testbed for studying coordination, pursuit–evasion, and emergent behavior in Multi-Agent Reinforcement Learning (MARL).

<h3>Early Environment Snapshot</h3>
<img src="assets/images/game_snap_v2.png" alt="Early Environment Snapshot" width="400"/>

---

## Overview

This framework provides a **discrete, grid-based simulation** designed to support controlled, interpretable, and reproducible experiments in MARL. It models classic predator-prey dynamics where multiple agents interact and learn in a bounded grid world.

Key goals:

* Facilitate **mechanistic understanding** of MARL behavior.
* Support **reproducible research** and ablation studies.
* Provide an **accessible learning tool** for students and new researchers.

---

## Features

### 🔍 Fully Interpretable
**State and action spaces are fully enumerable and transparent**, making it easy to track agent behavior, transitions, and environment evolution step-by-step.

### 🧩 Modular and Customizable
**Variables and reward structures can be modified in isolation**, enabling controlled experimentation. Easily change grid size, agent behavior, episode length, reward schemes, and terminal conditions.

### 🧪 Built for Rigorous Experimentation
**Reproducible ablation experiments** are supported by design. The codebase encourages transparency, logging, and clear interpretation of learning dynamics.

---

## Quick Start

### Installation

Clone the repository:

```bash
git clone https://github.com/ProValarous/Predator-Prey-Archetype-Gridworld-Environment.git
cd Predator-Prey-Archetype-Gridworld-Environment
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Minimal Example
```python
from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

# Define agents (type, team, name)
predator = Agent("predator", "predator_1", "Hunter")
prey = Agent("prey", "prey_1", "Runner")

# Create environment
env = GridWorldEnv(
    agents=[predator, prey],
    size=8,
    render_mode="human"
)

# Run a single episode
obs, info = env.reset(seed=42)
done = False

while not done:
    actions = {
        "Hunter": env.action_space.sample(),
        "Runner": env.action_space.sample()
    }
    result = env.step(actions)
    done = result["done"]

env.close()
```

## Citation

If you use this environment in your research, teaching, or project, please cite it using the following BibTeX:

```bibtex
@misc{predatorpreygridworld,
  author       = {Ahmed Atif},
  title        = {Predator-Prey Gridworld Environment},
  year         = {2025},
  howpublished = {\url{https://github.com/ProValarous/Predator-Prey-Gridworld-Environment}},
  note         = {A discrete testbed for studying Multi-Agent Reinforcement Learning dynamics.}
}
```

## License

This project is licensed under the [Apache-2.0 License](https://github.com/ProValarous/Predator-Prey-Gridworld-Environment/blob/main/LICENSE).