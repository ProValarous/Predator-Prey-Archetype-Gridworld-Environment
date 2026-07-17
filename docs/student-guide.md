# Student Wiki

This folder is your complete reading set for the 10-week program — a curated subset of the full wiki containing only what a student needs. Audit reports, architecture decision records, fix logs, and design history live in the parent `wiki/` and are not part of your workflow unless your instructor points you there.

---

## Folder Structure

```
students/
├── index.md               ← you are here
├── mission.md             ← why this project exists
│
├── overview/
│   ├── index.md           ← project scope and goals
│   ├── architecture.md    ← the four-layer design
│   └── glossary.md        ← term definitions
│
├── concepts/
│   ├── gridworld.md       ← the grid, obstacles, captures, termination
│   ├── agents.md          ← predators, prey, teams, identity
│   ├── observations.md    ← what agents perceive
│   ├── rewards.md         ← how reward signals are computed
│   └── marl.md            ← multi-agent RL theory
│
├── flows/
│   ├── init-flow.md       ← from YAML to wired environment
│   ├── step-flow.md       ← inside env.step()
│   └── training-loop.md   ← episode loop and Q-table update
│
├── guides/
│   ├── quickstart.md      ← install + first run
│   ├── custom-observation.md
│   ├── custom-reward.md
│   ├── config-recipes.md
│   └── using-evaluate.md
│
├── specs/
│   ├── observation-builder-spec.md
│   ├── reward-function-spec.md
│   └── algorithm-spec.md
│
└── reference/
    ├── api-reference.md
    └── config-reference.md
```

---

## Day 1 Reading Order

Read these in order before doing anything else.

1. [mission.md](mission.md) — what this project is, what it is not, and why it exists
2. [overview/architecture.md](overview/architecture.md) — the four layers and what each one does
3. [overview/glossary.md](overview/glossary.md) — terms you will see throughout the code and wiki
4. [guides/quickstart.md](guides/quickstart.md) — install dependencies and run your first training job

---

## Week-by-Week Reading Guide

| Week | Read | Purpose |
|------|------|---------|
| 1 | [concepts/gridworld.md](concepts/gridworld.md) · [concepts/agents.md](concepts/agents.md) | Understand what the env simulates |
| 1 | [flows/init-flow.md](flows/init-flow.md) · [flows/step-flow.md](flows/step-flow.md) | Trace the code path before writing any |
| 1 | [flows/training-loop.md](flows/training-loop.md) | Understand the Bellman update and epsilon decay |
| 1–2 | [concepts/observations.md](concepts/observations.md) · [concepts/rewards.md](concepts/rewards.md) | Before touching IQL/CQL experiments |
| 1–2 | [concepts/marl.md](concepts/marl.md) | Single-agent MDP → Markov game transition |
| 2–4 | [specs/algorithm-spec.md](specs/algorithm-spec.md) | Before implementing IDQN |
| 2–4 | [guides/config-recipes.md](guides/config-recipes.md) | Setting up ablation experiments |
| 2–4 | [guides/using-evaluate.md](guides/using-evaluate.md) | Collecting and interpreting metrics |
| Any | [guides/custom-observation.md](guides/custom-observation.md) | When adding a new observation builder |
| Any | [guides/custom-reward.md](guides/custom-reward.md) | When adding a new reward function |
| Any | [reference/config-reference.md](reference/config-reference.md) | When a YAML key is unclear |
| Any | [reference/api-reference.md](reference/api-reference.md) | When a method signature is unclear |

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

**I want to implement a new learning algorithm (DQN, PPO, Nash Q)**
→ [specs/algorithm-spec.md](specs/algorithm-spec.md) → [flows/training-loop.md](flows/training-loop.md) → [reference/api-reference.md](reference/api-reference.md)

**I want to change experiment parameters**
→ [reference/config-reference.md](reference/config-reference.md) → [guides/config-recipes.md](guides/config-recipes.md)

**I want to evaluate a trained policy**
→ [guides/using-evaluate.md](guides/using-evaluate.md)

**I want to understand the MARL theory behind what I am building**
→ [concepts/marl.md](concepts/marl.md) → [overview/architecture.md](overview/architecture.md)

---

## What Is Not in This Folder

| Excluded | Where it lives | Why excluded |
|----------|---------------|--------------|
| `decisions/ADR-*.md` | `wiki/decisions/` | Architecture history for maintainers |
| `reviews/audit-*.md` | `wiki/reviews/` | Bug audits assigned by instructor |
| `reviews/consistency-*.md` | `wiki/reviews/` | Wiki self-audits, maintainer use |
| `log.md` | `wiki/log.md` | Development journal, not actionable |
| `roadmap.md` | `wiki/roadmap.md` | Strategic context, not weekly-actionable |
| `program.md` | `wiki/program.md` | Instructor-facing program strategy |
| `troubleshooting.md` | `wiki/troubleshooting.md` | Kept in parent wiki for broader audience |
| `gotchas.md` | `wiki/gotchas.md` | Kept in parent wiki for broader audience |
| `specs/test-suite.md` | `wiki/specs/` | Test coverage map, maintainer reference |
| `weekly/week-01.md` | `wiki/weekly/` | Instructor distributes week tasks separately |

If a task requires one of these files, your instructor will point you there directly.
