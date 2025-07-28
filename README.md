<h1>🐾 Predator–Prey Gridworld Environment</h1>
  <p>A minimalist, discrete multi-agent predator-prey archytype environment.</p>

<h3> Early Environment SnapShot</h3>
<img src="miscellenous/imgs/game_snap.png" alt="Early Environment SnapShot" width="400"/>

---

## Overview

This repository provides a **discrete, grid-based predator-prey simulation environment** designed to support controlled, interpretable, and reproducible experiments in MARL. The environment models classic predator-prey dynamics where multiple agents (predators and prey) interact and learn in a bounded grid world.

Key goals of this framework:

* Facilitate **mechanistic understanding** of MARL behavior
* Support **reproducible research** and **ablation studies**
* Provide an **accessible learning tool** for students and new researchers

---

## Features

### $$ Fully Interpretable

* **State and action spaces are fully enumerable and transparent**, making it easy to track agent behavior, transitions, and environment evolution step-by-step.

### $$ Modular and Customizable

* **Variables and reward structures can be modified in isolation**, enabling controlled experimentation.
* Easily change grid size, agent behavior, episode length, reward schemes, terminal conditions, etc.

### $$ Ideal for Rigorous Experimentation

* **Reproducible ablation experiments** are supported by design.
* Codebase encourages transparency, logging, and clear interpretation of learning dynamics.

### $$ Education-Ready

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

### Installation

Clone the repository:

```bash
git clone https://github.com/ProValarous/Predator-Prey-Gridworld-Environment.git
cd Predator-Prey-Gridworld-Environment
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Example Usage

```python
from multi_agent_package.gridworld import GridWorldEnv
from multi_agent_package.agents import Agent

# Define agents
agent1 = Agent("prey", "Tom")
agent2 = Agent("predator", "Jerry")

# Create environment
env = GridWorldEnv(agents=[agent1, agent2], render=True)

# Run a single episode
obs = env.reset()
done = False
while not done:
    actions = env.sample_random_actions()
    obs, rewards, done, info = env.step(actions)
    env.render()
```

---

## Contributing

We welcome contributions that improve the clarity, utility, or extensibility of the environment. If you have ideas for enhancements, fixes, or new features, please open an issue or submit a pull request.

---

## Citation

If you use this environment in your research, teaching, or project, please cite it using the following BibTeX:

```bibtex
@misc{predatorpreygridworld,
  author       = {ProValarous},
  title        = {Predator-Prey Gridworld Environment},
  year         = {2025},
  howpublished = {\url{https://github.com/ProValarous/Predator-Prey-Gridworld-Environment}},
  note         = {A discrete testbed for studying Multi-Agent Reinforcement Learning dynamics.}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For any questions, issues, or collaborations, please reach out via the [GitHub repository](https://github.com/ProValarous/Predator-Prey-Gridworld-Environment/issues).

---

## Acknowledgements

This project draws inspiration from classic RL environments and aims to provide a more transparent, MARL-specific framework. Contributions from the community are deeply appreciated!
