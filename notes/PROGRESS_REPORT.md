# Predator-Prey MARL Testbed — Progress Report

**Project:** Predator-Prey Archetype Gridworld Environment
**Branch:** `feat/actor-critic-baseline` → merged with `feat/a2c-integration`
**Date:** 22 July 2026 (updated 23 July 2026 — A2C integration + verification added)

---

## ActorCritic Baseline (On-Policy)

### Implementation

Added a second baseline family alongside DQN: `ActorCritic`, an independent, **on-policy** algorithm (Sutton & Barto, *Reinforcement Learning: An Introduction* 2nd ed., Algorithm 13.5 — one-step online actor-critic). One `ActorCriticNetwork` per agent (shared trunk, a policy head producing action logits and a value head producing a scalar state-value), one optimizer per agent, following the same independent-per-agent pattern as DQN — but no replay buffer and no target network. Every single environment step immediately triggers one gradient update from that step's TD(0) error; exploration comes from sampling the stochastic policy directly rather than an epsilon schedule.

Registered as `actor_critic`, wired into `configs/experiment_actor_critic.yaml` and `scripts/run_actor_critic.py`, same config-driven pattern as every other baseline. 21 unit tests added; full suite (344 tests) still passes.

### First Verification Run — Baseline (Pre-Fix)

Ran the shared 3-predator-vs-3-prey config as a first real check that the algorithm learns as intended (not just that it executes without error): 10×10 grid, 20% obstacles, predator speed 1 / prey speed 3, 1000 episodes, `hidden_layers=[128,128]`, `learning_rate=0.001`, `gamma=0.99`, seed 42.

#### Average Reward by Training Quarter (averaged across the 3 agents per species)

| Species | Q1 | Q2 | Q3 | Q4 |
|---------|:--:|:--:|:--:|:--:|
| Predator | −2216 | −2102 | −2032 | −2235 |
| Prey | 299 | 282 | 272 | 299 |

#### Average Critic Loss by Training Quarter (averaged across the 3 agents per species)

| Species | Q1 | Q4 |
|---------|:--:|:--:|
| Predator | ~687,000 | ~602,000 |
| Prey | ~14,800 | ~14,400 |

**Issue found: the critic loss is enormous and not shrinking.** Root cause: the critic used plain MSE (`delta.pow(2)`), whereas DQN deliberately uses **Huber loss** (`SmoothL1Loss`) specifically because it is robust to large TD errors — and this config's rewards accumulate into the thousands per episode. Squared error on targets that large produces gradients large enough to destabilize the shared trunk (which both the policy and value heads depend on).

### After the Huber Loss Fix — Same Config, Re-run

| | Predator reward (Q1→Q4) | Prey reward (Q1→Q4) | Predator loss (Q1→Q4) | Prey loss (Q1→Q4) |
|---|---|---|---|---|
| **Before (MSE)** | −2216 → −2235 (peaks −2032 at Q3) | 299 → 299 (flat) | ~687,000 → ~602,000 | ~14,800 → ~14,400 |
| **After (Huber)** | −2631 → −2547 (flat/noisy) | 348 → 339 (flat) | ~211 → ~224 | ~32 → ~35 |

**The loss fix worked exactly as diagnosed.** Predator loss dropped roughly 3 orders of magnitude, confirming the diagnosis rather than reflecting run-to-run noise. **Reward still did not show a clean learning trend for either species.** Open question at this point: is the flat reward curve a genuinely hard task (3v3, predators 3× slower than prey) or an under-tuned algorithm?

### 1v1 Sanity Check — Full 2000-Episode Run

To separate "hard task" from "algorithm needs tuning," re-ran on `configs/dqn_1v1` — 1v1, predator speed 2 / prey speed 1, `hidden_layers=[64,64]` — the exact config DQN already has real training data on (`training_curves_dqn_1v1.csv`), so results are directly comparable. Matched DQN's own 2000-episode horizon (its config notes *"500 was not enough; 2000 gives DQN time to converge"*).

#### ActorCritic vs. DQN, identical config, identical episode count

| | Predator reward Q1 | Predator reward Q4 | Prey reward Q1 | Prey reward Q4 |
|---|:--:|:--:|:--:|:--:|
| **DQN** | −970 | −2078 | −92 | −64 |
| **ActorCritic** (entropy_coef=0) | −4135 | −4208 | −6.4 | −0.6 |

DQN shows a clear, monotonic *degradation* trend (co-evolutionary collapse — predator gets worse as prey adapts, consistent with the wider speed-sweep pattern). ActorCritic's reward is flat across all four quarters and starts already far worse than DQN's Q1.

#### The real signal: capture rate (fraction of episodes ending in a capture)

| | Q1 | Q2 | Q3 | Q4 |
|---|:--:|:--:|:--:|:--:|
| Capture rate | 6.4% | 5.6% | 3.4% | **0.6%** |

By decile (finer view, confirms this is a real trend, not quartile noise): `9%, 3%, 7%, 6.5%, 4.5%, 5%, 1.5%, 2%, 1%, 0.5%`.

**This settles the task-difficulty-vs-tuning question: it's tuning.** Capture rate isn't flat, it's *declining* — the predator is getting systematically worse at the actual objective over training, not just failing to improve. Meanwhile the critic's loss drops smoothly (20.4 → 0.33 across quarters) — in isolation that looks like convergence, but combined with the collapsing capture rate it's the signature of a specific failure mode: the critic is learning to predict a *boring, predictable* outcome (episode times out, no capture, steady shaping penalty) because the policy has stopped attempting anything else. Low surprise = low loss, even as the underlying behavior gets worse.

**Diagnosis: `entropy_coef=0.0` gives the policy no counter-pressure against this collapse.** DQN's own config for this exact scenario deliberately slows epsilon decay so both agents "explore together longer" through nearly the whole 2000-episode run. ActorCritic currently has no equivalent mechanism — once the stochastic policy's entropy collapses, nothing pulls it back toward exploring further.

### Entropy Regularization — Confirmed Fix

Re-ran the identical 1v1 config with `entropy_coef` raised from `0.0` to `0.01` — the standard default in the A2C/A3C literature — same episode count, same seed, only that one parameter changed.

| | Capture rate Q1 | Capture rate Q4 | Predator reward Q1 | Predator reward Q4 |
|---|:--:|:--:|:--:|:--:|
| `entropy_coef=0.0` | 6.4% | **0.6%** (collapsing) | −4135 | −4208 |
| `entropy_coef=0.01` | 35.6% | **36.4%** (stable) | −3447 | −3289 |

By decile with entropy on: `36.5%, 37%, 33.5%, 34.5%, 36.5%, 33.5%, 34%, 36%, 35%, 37.5%` — flat throughout, no late-training decay anywhere in the run. Overall capture rate went from 4% to 35.4%, roughly a 9× improvement, and the collapse pattern is gone entirely.

**This confirms the diagnosis.** Restoring exploration pressure directly fixed the capture-rate collapse, with only one parameter changed between runs.

**One nuance, kept honest rather than smoothed over:** predator *reward* itself stays deeply negative and roughly flat (~−3300 to −3450) even with the much higher capture rate — the accumulated per-step distance-shaping penalty across the ~65% of episodes that still end in timeout dominates the reward signal, so "reward improving" and "actual task performance improving" are not the same story here. Capture rate is the metric that actually reflects what changed; raw reward does not yet show it clearly.

### Next Steps

1. ~~Update the default `entropy_coef` in `configs/experiment_actor_critic.yaml`~~ — done, now `0.01`.
2. ~~Re-test the harder 3v3 config with `entropy_coef=0.01`~~ — done, see below.
3. **Carry this forward into A2C** — start its config with `entropy_coef=0.01` from day one rather than rediscovering the same collapse.
4. **Investigate the reward-vs-capture-rate disconnect** separately — likely a reward-shaping scale question (the distance penalty may be sized for a different capture rate regime), not an urgent blocker.

### 3v3 Retest with `entropy_coef=0.01`

Re-ran the original 3-predator-vs-3-prey config (the one from the very first verification run) with the new `entropy_coef=0.01` default, same 1000 episodes.

#### Aggregate reward/loss — entropy=0.0 vs entropy=0.01, both post-Huber-fix

| | Predator reward Q1→Q4 | Prey reward Q1→Q4 | Predator loss Q1→Q4 |
|---|---|---|---|
| `entropy_coef=0.0` | −2631 → −2547 | 348 → 339 | 211 → 224 |
| `entropy_coef=0.01` | −2358 → −2519 | 323 → 339 | 239 → 215 |

**Nearly identical.** Unlike 1v1, entropy regularization does not visibly move the aggregate reward/loss numbers on this config.

#### But the raw log tells a different story than the aggregates

Counted actual capture events from the console log's `steps=` field (a full non-capture episode is 167 steps here — prey speed 3 makes `SpeedWrapper` run 3 sub-steps per logical step, so 500/3≈167 — a capture ends the episode early, same signal used to spot captures in the 1v1 run). Across the 100 logged checkpoints, captures occurred in roughly 16% of Q1 samples, 20% of Q2, 28% of Q3, 20% of Q4 — steady throughout, no collapse. All three predators land captures at different points across the run (not one agent dominating), and the reward design correctly discriminates: the catching predator gets a clear positive reward (+60 to +95) while its two packmates take a small shaping-penalty hit that episode.

**Honest gap in the evidence:** the original `entropy_coef=0.0` 3v3 run was only analyzed via reward/loss aggregates, not the log's `steps=` field, so there is no equivalent capture-rate breakdown to compare against for that run. It cannot be said with confidence whether `entropy_coef=0.01` improved 3v3 capture behavior versus `0.0`, only that captures are not collapsing at `0.01`.

**Conclusion:** the entropy fix is cleanly validated on 1v1 (dramatic, well-evidenced before/after). On 3v3 it is not hurting — aggregate metrics are essentially a wash, and captures happen steadily rather than collapsing — but calling it "confirmed to help 3v3" would overstate what was actually measured. Reporting this as *validated on 1v1, directionally consistent but not conclusively confirmed on 3v3* rather than a stronger claim.

---

## Status: AC baseline ready to commit

Implementation, Huber-loss fix, entropy-regularization fix, and both verification runs (1v1 and 3v3) are complete and documented above. Remaining before commit: revert local `device: "cuda"` overrides back to `"cpu"` in both `configs/experiment_actor_critic.yaml` and `configs/dqn_1v1/experiment_actor_critic.yaml` (CI runs CPU-only), and a final full-suite pass. A2C is a separate track — merging a teammate's existing implementation into its own branch, then combining with this AC branch before A3C.

---

## A2C Integration (Areesha's implementation, PR #47)

### Merge

Pulled Areesha's `A2C` branch (`src/ppage/baselines/A2C/`: separate actor + critic networks per agent, on-policy `n_steps`-rollout updates) into `feat/a2c-integration`, cut from the finished AC branch. Two conflicts, both purely additive (`src/ppage/baselines/__init__.py` registry import, `src/ppage/baselines/README.md` directory/algorithm sections) — resolved by keeping both entries. Her actual algorithm code merged with zero conflicts. Combined suite: 376 tests passing (344 AC/existing + 32 hers).

Also fixed pre-existing Black/flake8 findings in her new files (`a2c.py`, `actor_network.py`, `critic_network.py`, `run_a2c.py`, `test_baselines_a2c.py`) — unlike the earlier STRP-wide formatting gap, these were new-to-this-branch issues, not pre-existing upstream debt, so fixed directly rather than just flagged.

### Pre-PR Verification

Same reasoning as AC: passing unit tests don't guarantee correct learning behavior — that lesson came from AC's own critic-loss and entropy bugs, neither of which any test caught. Her config's own comment (`# weight of the critic's MSE loss`) was a specific, testable prediction that A2C likely had the same critic-loss-scale issue AC did, so verified before opening a PR rather than after.

#### Critic loss — confirmed and fixed

Ran `configs/dqn_1v1/experiment_a2c.yaml` (2000 episodes, same config DQN/AC already have data on). Predator critic loss sat at 262,519 → 191,375 across quarters — hundreds of thousands, same magnitude and non-shrinking pattern as AC's pre-fix numbers. Root cause: identical to AC — plain `nn.MSELoss()` on this environment's large reward scale. Fix: swapped to `nn.SmoothL1Loss()` (Huber), matching AC/DQN.

| | Predator critic loss Q1→Q4 | Prey critic loss Q1→Q4 | Capture rate Q1→Q4 |
|---|---|---|---|
| Before (MSE) | 262,519 → 191,375 | 630 → 413 | 18.8% → 15.6% |
| After (Huber, entropy_coef=0.01) | 144 → 154 | 5.7 → 6.5 | 22.8% → 22.4% |

~1,500× reduction in critic loss, and capture rate came out slightly higher and more stable across quarters as a side effect (cleaner advantage estimates from a saner critic).

Also brought her console log format in line with AC's (`Episode X/Y | steps=Z | rewards: ... | avg_*_loss: ...` structure), extended to her three tracked metrics (`avg_critic_loss`, `avg_actor_loss`, `avg_entropy`) rather than collapsed to one number, since she already tracks more than AC does.

#### Entropy collapse — investigated, not solved, flagged as a follow-up

The same 1v1 run also revealed `predator_1_entropy` collapsing to ~0 by Q2 (`0.149 → 0.004 → 0.004 → 0.001`) despite `entropy_coef=0.01` already being set from the start — a more severe manifestation than AC ever showed (AC's entropy-fixed run held a stable ~35% capture rate; this collapses anyway). Checked her actor-loss formula directly — mechanically correct, same sign convention as AC's, no implementation bug found.

Tried raising `entropy_coef` to `0.05` (5×) on the identical config: the entropy metric **still** collapses to ~0 by Q3/Q4 (`0.1715 → 0.0003 → 0.0 → 0.0`) — raising the coefficient does not prevent the collapse, meaning this is likely a structural training-dynamics issue (e.g. logit magnitudes growing large enough that the softmax saturates, at which point the entropy bonus's weight stops mattering much) rather than a simple coefficient-magnitude problem.

**However, capture rate still improved substantially anyway:**

| | Capture rate Q1→Q4 |
|---|---|
| entropy_coef=0.01 | 22.8% / 21.0% / 22.4% / 22.4% |
| entropy_coef=0.05 | 32.2% / 29.2% / 31.0% / 28.8% |

So `entropy_coef` default raised to `0.05` in `configs/experiment_a2c.yaml` — it doesn't fix the underlying collapse, but it measurably and consistently helps task performance regardless. The root cause of the collapse itself is an open question, documented here for Areesha (or whoever picks it up) rather than solved in this branch.

### Status: A2C integration ready to commit

Merge conflicts resolved, lint clean, critic-loss fix applied and verified, entropy-coef default raised (partial fix, documented limitation), console logging made consistent with AC. Remaining before commit: revert local `device: "cuda"` overrides back to `"cpu"` in `configs/dqn_1v1/experiment_a2c.yaml`, and a final full-suite pass.
