# Mission

## What This Project Is

The Predator-Prey Archetype Gridworld is a **discrete, deterministic multi-agent reinforcement learning environment** built as a controlled research and teaching testbed.

It models classic predator-prey dynamics on a bounded grid. Multiple agents—predators who pursue, prey who evade—interact under configurable observation models and reward structures.

---

## Why It Exists

Most MARL benchmarks optimize for difficulty or realism. This project optimizes for **interpretability and scientific control**.

The core claim: to understand *why* a multi-agent system behaves a certain way, you need to be able to isolate variables. You need to know that when you change the reward structure, only the reward changed. When you change the observation model, only the perception changed. Nothing else moved.

This requires:
- **Determinism**: same seed → same trajectory, always
- **Modularity**: observation, reward, and learning are independently swappable
- **Immutability**: the physics never change between experiments
- **Config-driven variation**: ablation conditions are YAML files, not code branches

---

## What It Optimizes For

In priority order:

1. **Reproducibility** — experiments are fully reproducible from a seed + config directory
2. **Mechanistic transparency** — every state, transition, and reward is traceable
3. **Research modularity** — swap one component without touching any other
4. **Teaching clarity** — a student can read any module and understand it in one sitting
5. **Extensibility** — new observation/reward/algorithm types require no core changes

Performance (wall-clock speed, sample efficiency, scale) is explicitly *not* a priority. This is a testbed, not a production training platform.

---

## What It Is Not

- A high-performance RL library (use RLlib, SB3, or CleanRL for that)
- A continuous-space environment (all state/action spaces are discrete)
- A deep RL testbed out of the box (tabular baselines only; Gymnasium-compatible for wrapping)
- A photorealistic simulator

---

## Success Criteria

The project succeeds when:
- A researcher can run a 3-condition ablation study (observation type × reward shaping × algorithm) and get fully comparable, reproducible results purely by changing YAML files
- A student can add a custom reward function in under 30 minutes without modifying any existing file
- The physics of the gridworld are stable enough that a paper citing this repo can be reproduced 2 years later
