# Algorithms Overview

This project ships seven learning **baselines**, all living under
`src/baselines/` and all talking to the environment only through
`env.reset()` / `env.step()`. Three are tabular (they store a Q-table), one is a
value-based neural network (DQN), and three are policy-gradient actor-critic
methods that differ in *how* they update (online, batched, or asynchronous).
This page helps you pick one; each has its own deep-dive.

Before reading these, make sure you are comfortable with
[RL Foundations](../concepts/rl-foundations.md) (Q-learning, the Bellman equation)
and [MARL Theory](../concepts/marl.md) (what changes with many agents).

## The seven baselines

| Algorithm | Kind | Coordination | Deep-dive |
| --- | --- | --- | --- |
| **IQL** — Independent Q-Learning | tabular | none — each agent learns alone | [IQL](iql.md) |
| **CQL** — Centralized Q-Learning | tabular | full — one joint Q-table | [CQL & MixedTrainer](cql-mixed.md) |
| **MixedTrainer** | tabular | per-team (IQL or CQL each) | [CQL & MixedTrainer](cql-mixed.md) |
| **DQN** — Deep Q-Network | neural | none — one network per agent | [DQN](dqn.md) → [Variants](../concepts/dqn-variants.md) |
| **ActorCritic** — one-step online Actor-Critic | neural, on-policy | none — one network per agent | [Actor-Critic](actor-critic.md) |
| **A2C** — Advantage Actor-Critic | neural, on-policy, batched | none — one actor+critic per agent | [A2C](a2c.md) |
| **A3C** — Asynchronous Advantage Actor-Critic | neural, on-policy, async | none — shared network per agent, multi-process | [A3C](a3c.md) |

The first four are **value-based** (they learn Q-values and act greedily). The
three actor-critic variants all learn a stochastic policy directly, alongside a
state-value critic used only to reduce gradient variance — they differ in when
and how updates happen: **ActorCritic** updates once per environment step, no
buffer; **A2C** accumulates a short rollout before one batched update; **A3C**
runs multiple worker processes, each updating a *shared* network asynchronously,
lock-free (Hogwild-style). SAC is **not implemented** — see
[Roadmap](#roadmap) below.

## Which one should I use?

```mermaid
flowchart TD
    Q1{"State space small enough<br/>to enumerate in a table?"}
    Q1 -->|"no (large obs, want generalization)"| DQN["DQN<br/>(neural, per-agent)"]
    Q1 -->|yes| Q2{"Do agents need to<br/>coordinate explicitly?"}
    Q2 -->|"no / simplest baseline"| IQL["IQL"]
    Q2 -->|"yes, one team as a unit"| CQL["CQL"]
    Q2 -->|"different per team"| MIX["MixedTrainer<br/>(e.g. CQL predators vs IQL prey)"]
```

Rules of thumb:

- **Start with IQL.** It is the simplest correct baseline and trains fastest.
- **Use CQL** when a team should be treated as one decision-maker — but watch the
  cost: its table grows as `action_dim ^ n_agents`, so keep grids and agent counts
  small.
- **Use MixedTrainer** to study asymmetry, e.g. centralized predators against
  independent prey (exactly the kind of comparison in the
  [research study](../reference/papers.md) built on this environment).
- **Use DQN** when the observation is too large to tabulate or you want a policy
  that generalizes across states; enable [Double/Dueling](../concepts/dqn-variants.md)
  for a stronger variant.
- **Use ActorCritic** when you want a directly-learned stochastic policy instead
  of a greedy-over-Q one, or as the on-policy counterpart to compare against DQN
  in a comparative study.
- **Use A2C** when you want the same idea as ActorCritic but with batched,
  lower-variance updates instead of noisy per-step ones.
- **Use A3C** when you want to study asynchronous, multi-worker training
  dynamics specifically — it's CPU-parallel, not GPU-accelerated, so it won't
  necessarily train faster wall-clock than A2C on a small gridworld; the point
  is the decorrelated, asynchronous updates themselves.

## The shared training contract

Every baseline subclasses `BaseAlgorithm` (`src/baselines/base.py`) and implements:

- `select_actions(observations) -> {agent: action}` — usually ε-greedy over Q.
- `train()` — the episode loop (see [Training Loop](../flows/training-loop.md)).
- `save(path)` / `load(env, config, path)` — checkpointing.
- `evaluate(episodes)` — greedy rollout returning mean episode length and per-agent
  return.

They are wired in by name through the algorithm registry
(`experiment.yaml → algorithm.name`), so switching algorithms is a one-line config
change.

## Roadmap

`ActorCritic` is the one-step **online** variant (Sutton & Barto, Algorithm
13.5) — no rollout buffer, one gradient update per env step. `A2C` batches a
short rollout before updating. `A3C` (Mnih et al., 2016) runs multiple worker
processes updating a shared network asynchronously. The following is
documented for context but **not implemented**:

- **SAC** — Soft Actor-Critic (Haarnoja et al., 2018); a discrete-action variant
  would be needed for this environment.

See [Papers & Further Reading](../reference/papers.md) for full citations.
