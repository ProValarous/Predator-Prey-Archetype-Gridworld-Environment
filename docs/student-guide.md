# Student Reading Guide

This page is the curated reading path through the documentation for students
joining the project. Everything it links to lives in this docs site; read in
the suggested order rather than alphabetically.

---

## What's Here

```
docs/
├── index.md                  ← docs home
├── mission.md                ← why this project exists
├── student-guide.md          ← you are here
│
├── overview/                 ← scope, architecture, glossary, roadmap
├── concepts/                 ← gridworld, agents, observations, rewards,
│                                actions, wrappers, MARL and RL foundations
├── flows/                     ← init flow, step flow, training loop
├── algorithms/                ← deep dives: IQL, CQL/Mixed, DQN, AC, A2C, A3C
├── guides/                    ← quickstart, custom plug-ins, config recipes,
│                                using evaluate
├── tutorials/                 ← first-experiment walkthrough
├── specs/                     ← plug-in and algorithm contracts
├── reference/                 ← API reference, config reference, papers
└── api/                       ← auto-generated API pages (mkdocstrings)
```

---

## Day 1 Reading Order

Read these in order before doing anything else.

1. [mission.md](mission.md) — what this project is, what it is not, and why it exists
2. [overview/architecture.md](overview/architecture.md) — the five layers (core, plugins, wrapper, baselines, orchestration) and what each one does
3. [overview/glossary.md](overview/glossary.md) — terms you will see throughout the code and docs
4. [guides/quickstart.md](guides/quickstart.md) — install dependencies and run your first training job
5. [tutorials/first-experiment.md](tutorials/first-experiment.md) — a guided first experiment, end to end

---

## Week-by-Week Reading Guide

| Week | Read | Purpose |
|------|------|---------|
| 1 | [concepts/rl-foundations.md](concepts/rl-foundations.md) | The RL vocabulary everything else assumes |
| 1 | [concepts/gridworld.md](concepts/gridworld.md) · [concepts/agents.md](concepts/agents.md) | Understand what the env simulates |
| 1 | [flows/init-flow.md](flows/init-flow.md) · [flows/step-flow.md](flows/step-flow.md) | Trace the code path before writing any |
| 1 | [flows/training-loop.md](flows/training-loop.md) | Understand the Bellman update and epsilon decay |
| 1–2 | [concepts/observations.md](concepts/observations.md) · [concepts/rewards.md](concepts/rewards.md) · [concepts/actions.md](concepts/actions.md) | Before touching IQL/CQL experiments |
| 1–2 | [concepts/marl.md](concepts/marl.md) | Single-agent MDP → Markov game transition |
| 2–4 | [algorithms/index.md](algorithms/index.md) and the deep-dive for your algorithm | Before implementing or modifying a baseline |
| 2–4 | [specs/algorithm-spec.md](specs/algorithm-spec.md) | The contract a new algorithm must satisfy |
| 2–4 | [guides/config-recipes.md](guides/config-recipes.md) | Setting up ablation experiments |
| 2–4 | [guides/using-evaluate.md](guides/using-evaluate.md) | Collecting and interpreting metrics |
| Any | [guides/custom-observation.md](guides/custom-observation.md) | When adding a new observation builder |
| Any | [guides/custom-reward.md](guides/custom-reward.md) | When adding a new reward function |
| Any | [guides/custom-action.md](guides/custom-action.md) | When adding a new action space |
| Any | [concepts/wrappers.md](concepts/wrappers.md) | When speed/stamina mechanics matter to your experiment |
| Any | [reference/config-reference.md](reference/config-reference.md) | When a YAML key is unclear |
| Any | [reference/api-reference.md](reference/api-reference.md) | When a method signature is unclear |
| Any | [reference/papers.md](reference/papers.md) | The literature behind each algorithm |

---

## Navigation by Task

**I want to run the environment for the first time**
→ [guides/quickstart.md](guides/quickstart.md)

**I want to understand what my algorithm is receiving**
→ [concepts/observations.md](concepts/observations.md) → [flows/step-flow.md](flows/step-flow.md)

**I want to understand why an agent gets a certain reward**
→ [concepts/rewards.md](concepts/rewards.md) → [flows/training-loop.md](flows/training-loop.md)

**I want to add a new observation builder**
→ [guides/custom-observation.md](guides/custom-observation.md) → [specs/observation-builder-spec.md](specs/observation-builder-spec.md)

**I want to add a new reward function**
→ [guides/custom-reward.md](guides/custom-reward.md) → [specs/reward-function-spec.md](specs/reward-function-spec.md)

**I want to add a new action space**
→ [guides/custom-action.md](guides/custom-action.md) → [specs/action-space-spec.md](specs/action-space-spec.md)

**I want to implement a new learning algorithm (PPO, Nash Q, ...)**
→ [specs/algorithm-spec.md](specs/algorithm-spec.md) → [flows/training-loop.md](flows/training-loop.md) → [reference/api-reference.md](reference/api-reference.md)

**I want to change experiment parameters**
→ [reference/config-reference.md](reference/config-reference.md) → [guides/config-recipes.md](guides/config-recipes.md)

**I want to evaluate a trained policy**
→ [guides/using-evaluate.md](guides/using-evaluate.md)

**I want to understand the MARL theory behind what I am building**
→ [concepts/marl.md](concepts/marl.md) → [overview/architecture.md](overview/architecture.md)

---

## How to contribute what you build

When your reward function, observation builder, or experiment is ready to
share, follow [contributing.md](contributing.md) and
[git-workflow.md](git-workflow.md): branch off `STRP`, keep `core/` untouched,
and open your PR against `STRP`.
