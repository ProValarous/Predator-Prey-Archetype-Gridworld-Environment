# Concept: Rewards

## What a Reward Signal Is

A reward signal maps the current environment state to a per-agent scalar. It is the **incentive structure** of the experiment — the thing that tells agents what behavior is desirable.

This system separates rewards into two layers:
1. **Base rewards** — hardcoded physics penalties/bonuses baked into the environment
2. **Shaping rewards** — pluggable, configurable signals that layer on top

---

## Base Rewards (Hardcoded)

These values live in `GridWorldEnv.base_reward()` and cannot be changed via config:

| Event | Agent | Value |
|-------|-------|-------|
| Captures a prey (this step) | Predator | `+100.0` |
| Is captured (this step) | Prey | `−100.0` |
| Move blocked by obstacle | Any | `−200.0` |
| Each timestep | **Predator only** | `−5.0` |

The obstacle penalty (`−200`) exceeds the capture bonus (`+100`) to strongly discourage wall-hugging. The step cost (`−5`) applies **only to predators** — it incentivizes faster capture. Prey pay no per-step cost; the `SurvivalReward` shaper can be added to give prey a positive per-step signal instead.

`GridWorldEnv.step()` calls `self.base_reward()` **unconditionally, every step** — it is not routed through the plugin/registry system at all. You cannot disable or replace these signals without modifying core code.

> ⚠️ There is a `BaseReward` reward-function class (registry key `"base"`) that wraps `env.base_reward()` and scales it by `weight`. It is **not** part of the active reward chain — `run_from_config.build_environment()` deliberately does not add it, because `base_reward()` is already unconditionally applied inside `step()` (see above); chaining `BaseReward` on top would double-count every capture/step-cost/obstacle-penalty signal (this was a real bug, fixed in PR #26). `rewards.yaml`'s `base.enabled` flag is consequently **inert** post-fix: `run_from_config.py` reads it only to assert the key exists, never to gate anything — `base_reward()` runs regardless of what it's set to.

---

## Shaping Rewards (Pluggable)

Shaping rewards are `RewardFunction` subclasses that provide **dense guidance** toward desired behaviors. They do not replace base rewards; they are added on top.

### Why Shaping Matters

Without shaping, predators only receive signal when they capture prey (`+100`). In a 10×10 grid, a random agent rarely captures anything — the signal is so sparse that tabular Q-learning takes thousands of episodes to converge. Distance-based shaping provides gradient every step: moving closer → less negative reward.

### Available Shapers

| Key | Class | Signal | For |
|-----|-------|--------|-----|
| `predator_distance` | `PredatorDistanceReward` | `−weight × dist_to_nearest_prey` | Predators |
| `survival` | `SurvivalReward` | `+weight` per step alive | Prey |

---

## Reward Composition

Composition happens in two tiers:

```
total_reward[agent] = base_reward[agent]              ← computed directly by gridworld.step(),
                                                          always on, never via a plugin
                    + sum(shaping_fn.compute(env)[agent]
                          for shaping_fn in configured_shapers)  ← env.reward_fn, added on top
```

The shaping sum happens in a closure built by `run_from_config.py` and assigned to `env.reward_fn`; `gridworld.step()` adds its output onto the already-computed base reward (`rewards[k] += custom.get(k, 0.0)`). Adding a new **shaping** component means adding an entry to `rewards.yaml`'s `shaping` list — no code change needed. The base reward itself cannot be added/removed this way (see above).

**Limitation:** The current composition gives no visibility into individual components during training. If you need per-component logging, you must modify the closure or add a wrapper.

---

## Reward Design Trade-offs

| Trade-off | Consideration |
|-----------|--------------|
| Dense vs. sparse | Dense (shaping) accelerates learning but may cause reward hacking |
| Symmetric vs. asymmetric | Predator and prey rewards are on different scales (+100 capture vs +1 survival) |
| Base reward weight | Scaling base rewards via `weight` also scales the obstacle penalty and step cost |
| Shaping weight tuning | High `predator_distance` weight may cause predators to ignore obstacles |

---

## Reward as an Experiment Variable

The reward system is a **first-class experiment variable**. A common ablation study:

| Condition | rewards.yaml |
|-----------|-------------|
| Sparse only | `base` reward only |
| + Predator shaping | `base` + `predator_distance` |
| + Survival shaping | `base` + `survival` |
| Full shaping | `base` + `predator_distance` + `survival` |

Each condition is a YAML change. No code changes required.

---

## MARL Reward Considerations

### Credit Assignment

When multiple predators share a cell with a prey, **all** of them receive the full `+100.0` bonus — the reward is not divided. This is a deliberate choice: in independent Q-learning each agent's Q-table is updated separately, so splitting credit would require knowing which predator "caused" the capture, which is undefined in simultaneous movement.

The consequence: cooperative multi-predator strategies are rewarded more than solo captures (the team collectively earns more), which creates natural incentive alignment without any explicit coordination mechanism.

### Non-Stationarity

Because each agent's Q-table changes every episode, the effective reward signal seen by any single agent is **non-stationary**: the same observation can lead to different outcomes depending on what the other agents do. This violates the stationarity assumption that standard Q-learning convergence proofs rely on.

In practice: IQL often works but may oscillate or converge to suboptimal joint policies. See [concepts/marl.md](marl.md) for details.

### Competitive vs. Cooperative Dynamics

The predator-prey setup is **mixed competitive-cooperative**: predators cooperate implicitly (multiple predators in the same region increase capture probability) while competing with prey. Reward shaping must account for both sides:

- Predator shaping (e.g., `predator_distance`) speeds predator learning at the cost of potential reward hacking (predators orbit prey without capturing)
- Prey shaping (e.g., `survival`) gives prey a gradient to evade even in sparse-capture episodes

---

## The Reward Function Interface

Every reward function:
1. Inherits from `RewardFunction` in `rewards/base.py`
2. Implements `compute(env) → Dict[str, float]`
3. Accepts `weight` and optional kwargs at construction
4. Is **read-only** — must not modify env or agent state

For the full contract, see [specs/reward-function-spec.md](../specs/reward-function-spec.md).
