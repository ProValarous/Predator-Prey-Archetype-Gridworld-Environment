# 🐾 Predator–Prey Gridworld Environment

A **deterministic, modular, research-grade multi-agent predator–prey environment** built to study coordination, pursuit–evasion, and emergent behavior in Multi-Agent Reinforcement Learning (MARL).

This repository is not just a simulation.

It is a **controlled experimental laboratory** for understanding how multi-agent learning systems behave.

---

## 🎯 What This Repository Is About

This project provides:

* A discrete 2D gridworld with predators and prey
* Explicit, fully inspectable transition dynamics
* Pluggable observation models
* Pluggable reward functions
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

### 1️⃣ `multi_agent_package` — The Environment

Implements:

* Grid Environment dynamics
* Agent movement
* Capture logic
* Episode termination
* Observation plug-ins
* Reward plug-ins

This layer defines the world.

### 2️⃣ `baselines` — The Learning Algorithms

Implements:

* Independent Q-Learning (IQL)
* Centralized Q-Learning (CQL)
* MixedTrainer — per-team IQL/CQL assignment

This layer defines how agents learn.

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
├── baselines/                # Learning algorithms
└── multi_agent_package/      # Environment
    ├── core/                 # Immutable Environment dynamics
    ├── observations/         # Perception plug-ins
    ├── rewards/              # Incentive plug-ins
    ├── registry/             # Safe plug-in selection
    ├── scripts/              # Experiment runners

configs/                      # YAML experiment definitions
```

Core Environment dynamics is stable infrastructure.

Observations and rewards are the intended extension points.

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

```bash
git clone https://github.com/ProValarous/Predator-Prey-Archetype-Gridworld-Environment.git
cd Predator-Prey-Archetype-Gridworld-Environment
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m multi_agent_package.scripts.run_from_config
```

All experiments are launched from the repository root.

---

## 👩‍🎓 For Contributors and Students

You are encouraged to:

* Implement new reward functions
* Design new observation schemes
* Run structured experiments
* Perform reproducible ablations

You are not expected to modify core Environment dynamics.

This mirrors how research infrastructure is structured in practice.

---

## 📜 Citation

```bibtex
@misc{predatorpreygridworld,
  author       = {Ahmed Atif and contributors},
  title        = {Predator–Prey Gridworld Environment},
  year         = {2025},
  note         = {A deterministic modular testbed for Multi-Agent Reinforcement Learning}
}
```

---

## 📜 License

Apache License 2.0

---


