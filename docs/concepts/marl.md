# Concept: Multi-Agent Reinforcement Learning

This document explains the MARL-specific properties, limitations, and design choices of the predator-prey environment. It assumes familiarity with single-agent Q-learning.

---

## What Makes This Multi-Agent

In a standard single-agent MDP, one agent interacts with a stationary environment. Here, **multiple agents act simultaneously** — each predator and each prey takes an action every step. The environment state transitions depend on all actions together, not any single agent's action.

This creates three properties absent from single-agent RL:

1. **Joint action space** — the effective action each step is the Cartesian product of all agents' actions. With 3 predators and 2 prey, each taking 1 of 5 actions, there are 5⁵ = 3,125 joint actions per step.
2. **Partial observability** — each agent receives only its own observation, not the full joint state (unless `AbsoluteObservation` is used).
3. **Non-stationarity** — because all agents are learning simultaneously, the "environment" each agent faces is non-stationary: the same observation can lead to different outcomes as other agents' policies evolve.

---

## Independent Q-Learning (IQL)

IQL is the simplest MARL algorithm: each agent runs standard Q-learning independently, treating other agents as part of the environment.

```
For each agent i:
  Q_i(s_i, a_i) ← Q_i(s_i, a_i) + α [r_i + γ max_a Q_i(s_i', a) − Q_i(s_i, a_i)]
```

**What this ignores:** The transition `s_i → s_i'` depends on what all other agents did, not just agent `i`. As other agents learn, this distribution shifts — making the effective MDP non-stationary for agent `i`.

**In practice:** IQL often converges in small environments despite lacking formal guarantees. It is the simplest correct implementation and serves as the baseline of record here.

---

## Non-Stationarity

Non-stationarity is the central challenge of multi-agent learning. Consider predator `P1`:

- In episode 100, prey `R1` flees right. P1 learns to move right.
- In episode 200, prey `R1` has learned to flee left. P1's rightward policy is now wrong.
- P1 must re-learn — but by then, R1 may have changed again.

This cycle can prevent convergence or cause policy oscillation. It is fundamental to adversarial multi-agent settings (predator-prey is inherently adversarial).

**Mitigations not implemented:**
- Self-play with frozen opponents (train one side at a time)
- Population-based training
- Nash equilibrium solvers

---

## Convergence

IQL has **no convergence guarantee** in multi-agent settings. Single-agent Q-learning converges to the optimal policy under:
- Finite state-action space ✅
- Each state-action pair visited infinitely often ✅ (with sufficient exploration)
- Decaying learning rate ✅
- **Stationary environment** ❌ — violated because other agents are learning

For cooperative tasks (predators only, no adversary), IQL can converge because the effective environment becomes stationary once all agents converge. For adversarial predator-prey, convergence is not guaranteed.

---

## Credit Assignment

When predators `P1` and `P2` both occupy the same cell as prey `R1`:
- Both `P1` and `P2` are added to `_capturing_predators`
- Both receive `+100.0` reward
- Neither's Q-table distinguishes "I caused the capture" from "I was present for the capture"

This is the **credit assignment problem**: which agent deserves credit for a joint outcome? IQL resolves it trivially (everyone gets full credit), which:
- Creates incentive for predators to swarm together (capture is more likely)
- Does not incentivize individual skill or unique contribution
- May lead to degenerate strategies where all predators follow one another

---

## Centralized Training, Decentralized Execution (CTDE)

CTDE approaches (QMIX, MADDPG, VDN) train a centralized value function that sees all agents' observations and actions. At execution time, each agent still acts from its own observation only.

**This project does not implement CTDE.** See [ADR-004](../decisions/ADR-004-tabular-baselines.md).

Why: CTDE for tabular methods requires a joint state space of size `|S|^n` (exponential in agent count). For 5 agents on a 10×10 grid using `AbsoluteObservation`, the joint state space is infeasible.

---

## Exploration in Multi-Agent Settings

Each agent uses independent epsilon-greedy exploration. This means:
- All agents may explore simultaneously — no coordination of exploratory actions
- In cooperative tasks, uncoordinated exploration can be inefficient (two predators may explore in opposite directions)
- In adversarial tasks, prey exploring randomly may accidentally walk into a predator — providing misleading signal

No joint exploration strategy is implemented. Reducing epsilon faster for prey (they should exploit evasion quickly) or slower for predators (they need to explore the full grid) is possible via per-agent config but not currently supported.

---

## Summary: What IQL Can and Cannot Do Here

| Capability | IQL |
|-----------|-----|
| Learn to capture prey in sparse reward setting | ✅ (with enough episodes) |
| Coordinate multi-predator strategies | ⚠️ Implicitly (no explicit mechanism) |
| Converge to Nash equilibrium vs. learning prey | ❌ Not guaranteed |
| Generalize to unseen grid sizes | ❌ Tabular — cannot generalize |
| Handle captured agents cleanly | ⚠️ Continues updating their Q-tables unnecessarily |
| Distinguish credit among cooperating predators | ❌ All get full reward |

For a deeper treatment of MARL theory, see Shoham & Leyton-Brown (2009) *Multiagent Systems* or Lowe et al. (2017) *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*.
