# Code Audit: Bugs, Broken Code, and Architectural Violations

**Scope:** all of `src/` (core, plugins, wrappers, registries, baselines, scripts) on branch `STRP` (HEAD `070bdf3`).
**Method:** full read of every source file, cross-checked against the architecture rules in `CONTRIBUTING.md`, `docs/mission.md`, and `docs/overview/`. Two findings were verified by running the code (marked "verified"); the rest are confirmed by tracing the code.
**Not in scope:** tests were read only to confirm behavior, not audited for their own quality.

## Summary

| # | Severity | Area | Finding | Verdict |
|---|----------|------|---------|---------|
| 1 | HIGH | core | `allow_cell_sharing` is a dead config flag (agents always share cells) | CONFIRMED (verified) |
| 2 | HIGH | plugins | `local_radius` encode has unstable feature slots (issue #34, unfixed) | CONFIRMED (verified) |
| 3 | MEDIUM | baselines | terminated-vs-truncated conflation zeroes the bootstrap on timeout (all 4 algos) | CONFIRMED |
| 4 | MEDIUM | baselines | `BaseAlgorithm.evaluate()` collects/returns nothing; every `run_*.py --mode eval` prints no metrics | CONFIRMED |
| 5 | MEDIUM | baselines | `evaluate.py` evaluates an **untrained** algorithm (never loads/trains) | CONFIRMED |
| 6 | MEDIUM | scripts | `sweep.py` overwrites the tracked `configs/observations.yaml`, no `try/finally`, strips comments | CONFIRMED |
| 7 | MEDIUM | scripts | `render.py` runs full training, not a "single episode render" as documented | CONFIRMED |
| 8 | MEDIUM | config | `env.yaml` ships `render_mode: human` → default runs open a window and throttle to 4 fps | CONFIRMED |
| 9 | LOW | plugins | `SpeedWrapper.NOOP = 4` still hardcoded for idle sub-steps (residual of the #27 quirk) | CONFIRMED |
| 10 | LOW | plugins | obstacle keys sorted as strings (`obstacle_10` < `obstacle_2`) in several encoders | CONFIRMED |
| 11 | LOW | plugins | `action_registry` skips subclass validation, unlike the other two registries | CONFIRMED |
| 12 | LOW | plugins | `SpeedDiscreteActionSpace.to_moves()` is dead in production after #27 (tests only) | CONFIRMED |
| 13 | LOW | baselines | DQN target-sync cadence scales with agent count (`_train_steps` counts across agents) | CONFIRMED |
| 14 | LOW | core | agent speed/stamina defaults hardcoded in core (`predator=1, prey=3, stamina=10`) | CONFIRMED |
| 15 | LOW | core | duplicate action-decoding source (`agent._actions_to_directions` vs action plugin) | CONFIRMED |
| 16 | LOW | scripts | `scripts/test_baseline.py` collides with pytest collection; stale comment + dead param | CONFIRMED |
| 17 | LOW | scripts | `sweep.py` hardcodes `sweep("radius", [1,2,3,4])` | CONFIRMED |
| 18 | INFO | core | rendered "Up" moves visually down (y-down convention); base obstacle penalty rarely reachable | CONFIRMED |

**Architecturally clean (verified):** core imports nothing from plugins/baselines; all randomness flows through seeded `np.random.default_rng` / `torch.manual_seed`; registries are the wiring path; the base reward now has a single application path (the `BaseReward` plugin, #37).

---

## HIGH

### 1. `allow_cell_sharing` is a dead config flag
- **Where:** `src/multi_agent_package/core/gridworld.py:111` (stored), movement loop `:247-261` (never reads it); wired from config at `src/multi_agent_package/scripts/run_from_config.py:121`.
- **Category:** bug / config no-op (architectural).
- **What:** `self.allow_cell_sharing` is stored but never consulted anywhere in `step()`. Movement only checks `block_agents_by_obstacles`. Two agents can always occupy the same cell regardless of the flag.
- **Failure scenario (verified):** built an env with `allow_cell_sharing=False`, moved prey onto the predator's cell; both ended at `(2,2)`. The flag has zero effect.
- **Why it matters:** this is the exact bug class as the old #32 `base.enabled` no-op — a config key that silently does nothing and misleads users. It also quietly defines capture semantics (capture requires shared cells, which are always allowed).
- **Fix:** either implement the flag (reject a move whose target cell is occupied by another agent when `allow_cell_sharing=False`) or remove the flag and its config key and document that cell-sharing is always on.

### 2. `local_radius` observation has unstable feature slots (issue #34)
- **Where:** `src/multi_agent_package/observations/local_radius.py:64-96` (`encode`).
- **Category:** bug (correctness).
- **What:** `encode()` iterates `sorted(visible_agents.keys())`, appends the *visible* agents first, then zero-pads the remainder. An agent's feature slot therefore depends on **which** other agents are currently within radius, not on a fixed identity. Same issue for obstacles.
- **Failure scenario (verified):** predator sees only `prey_2` → its features land in slot 0; on another step it sees only `prey_1` → `prey_1` lands in the *same* slot 0. The encoder output is byte-identical for two different world states, so a DQN cannot tell the prey apart. Vector length stays fixed (so `DQN._validate_state_shape` does not catch it) — the corruption is silent.
- **Why it matters:** this is the only observation builder with this bug (the others always include every entity, so their sort is stable). It silently degrades any DQN run using `local_radius` (e.g. `configs/d3qn`, `configs/dqn_1v1` use `local_radius`).
- **Fix:** assign a fixed slot per identity — iterate over `sorted(a.agent_name for a in env.agents)` (self naturally stays zero) and over `range(len(env._obstacle_location))`, writing zeros for absent entities. (This is what the abandoned PR #33 attempted; extract that piece into a clean PR.)

---

## MEDIUM

### 3. Terminated-vs-truncated conflation cuts the value bootstrap on timeout
- **Where:** DQN `src/baselines/DQN/dqn.py:289` (and pushed as the done flag `:302`); IQL `src/baselines/IQL/iql.py:103,114`; CQL `src/baselines/CQL/cql.py:146,161`; MixedTrainer `src/baselines/MIXED/mix_train.py:205,217,233`.
- **Category:** RL-correctness.
- **What:** every algorithm computes `done = terminated or truncated` and uses it to zero the next-state value (`q_next = 0 if done`). Truncation (hitting `max_steps`) is not a terminal state — the episode is cut for time, not because the world ended — so bootstrapping should continue. Treating truncation as terminal teaches the agents that the states near the time limit have zero future value.
- **Failure scenario:** with `max_steps=500` and no capture, the final transition of every timed-out episode gets target `= reward + 0`, biasing `Q` downward for long-horizon states. Systematically distorts value estimates in exactly the regimes the project studies (pursuit that runs long).
- **Fix:** bootstrap on truncation. Push/derive a `terminal` flag that is `terminated` only (not `truncated`), and use that to gate the bootstrap; keep a separate `done` purely for ending the episode loop.

### 4. `BaseAlgorithm.evaluate()` computes nothing
- **Where:** `src/baselines/base.py:27-37`.
- **Category:** broken-code / incomplete.
- **What:** the method loops episodes and discards every result — no return value, no metric accumulation — despite the docs describing it as collecting episode length and per-agent return.
- **Failure scenario:** `run_iql.py`, `run_cql.py`, `run_mixed.py`, `run_dqn.py` all call `algo.evaluate()` in `--mode eval`. The command loads a checkpoint, runs episodes, and prints nothing — eval mode looks like it does nothing.
- **Fix:** make `evaluate()` accumulate and return episode-length + per-agent-return stats (as `scripts/evaluate.py` already does), and have the `run_*` scripts print the summary.

### 5. `evaluate.py` evaluates an untrained algorithm
- **Where:** `src/multi_agent_package/scripts/evaluate.py:26-30`.
- **Category:** bug / correctness-gap.
- **What:** it constructs a fresh algorithm and evaluates immediately — it never trains and never loads a checkpoint. Tabular agents evaluate on empty Q-tables (argmax of zeros → action 0); DQN evaluates on random weights with the configured (often exploratory) epsilon.
- **Failure scenario:** `python -m multi_agent_package.scripts.evaluate` prints "evaluation" metrics for a random policy, which reads as real results.
- **Fix:** add a `--load-path` and load the trained model before evaluating (and set epsilon to 0 for greedy eval).

### 6. `sweep.py` mutates a tracked config file unsafely
- **Where:** `src/multi_agent_package/scripts/sweep.py:14-26`.
- **Category:** bug / data-loss risk + architectural violation.
- **What:** `sweep()` rewrites `configs/observations.yaml` in place for each value, then restores the original only after the loop — with no `try/finally`. A crash (or Ctrl-C) mid-sweep leaves the user's config overwritten with a swept value. Even on success, `yaml.dump` rewrites the file without its comments and with reordered keys.
- **Failure scenario:** sweep over `radius=[1,2,3,4]`; the run for `radius=3` raises; `configs/observations.yaml` is left pinned at `radius=3` with all comments stripped.
- **Fix:** never mutate the shared tracked config. Build an in-memory config dict (or a temp config dir) and pass it to the runner, or wrap the restore in `try/finally`. Prefer isolated per-run configs (the project's own "config drives everything" principle).

### 7. `render.py` does not render a single episode
- **Where:** `src/multi_agent_package/scripts/render.py:1-8`.
- **Category:** broken-code (misleading).
- **What:** the docstring says "Render a single episode using a fixed seed," but the body calls `run_from_config.main("configs")`, which runs the **full default training experiment** (IQL, 500 episodes) using `env.yaml`'s `render_mode`.
- **Failure scenario:** running it launches a 500-episode training run, not a one-episode visualization; combined with finding 8 it renders every step at 4 fps and takes a very long time.
- **Fix:** implement an actual single-episode rollout (reset, step until done, render each step) driven by a loaded/looked-up policy or random actions, with `render_mode="human"` set locally.

### 8. Default config ships `render_mode: human`
- **Where:** `configs/env.yaml` (`render_mode: human`).
- **Category:** config footgun.
- **What:** the default env config opens a Pygame window and throttles every `step()` to `render_fps=4`. Any run that uses the default config without overriding it (`run_from_config`, `render.py`) is either blocked (no display, e.g. CI/headless) or crawls.
- **Fix:** default `render_mode: null`; set `human` only in a dedicated visualization path/config. (Every `run_*.py` already overrides to `None` except `run_from_config`/`render.py`, which is why those two are the slow ones.)

---

## LOW

### 9. `SpeedWrapper.NOOP = 4` still hardcoded
- `src/multi_agent_package/wrappers/speed.py:30`. PR #27 correctly routed the sub-step *budget* through `env.action_space_plugin.is_noop()`, but idle sub-steps are still filled with the literal action `4`. All three shipped action spaces use `4` as NOOP, so this works today; a custom action space with a different NOOP index would silently break sub-stepping. Fix: fill idle slots with the plugin's NOOP action rather than a constant.

### 10. Obstacle keys sorted as strings
- `observations/absolute.py:115`, `relative.py:129`, `local_radius.py:89`, `default.py:32`. Keys like `obstacle_10` sort before `obstacle_2`. It is consistent within a run (so not a learning bug), but the ordering is non-numeric and fragile if obstacle indices are ever reused/reordered. Fix: sort by integer index.

### 11. `action_registry` skips subclass validation
- `src/multi_agent_package/registry/action_registry.py:23-24`. `register_action_space` does not check `issubclass(cls, ActionSpace)`, unlike `reward_registry`/`observation_registry`. Documented, but an inconsistency that lets a bad class register silently.

### 12. `SpeedDiscreteActionSpace.to_moves()` is dead in production
- `src/multi_agent_package/actions/speed_discrete.py:13-18`. After #27, `SpeedWrapper` no longer calls `to_moves()`; only the test suite references it. Either wire it back or drop it (and its tests) to avoid a second, divergent sub-step-budget implementation.

### 13. DQN target-sync cadence scales with agent count
- `src/baselines/DQN/dqn.py:243-245`. `self._train_steps` increments once per agent per env step, so with N agents the target networks sync every `target_update_interval / N` env-steps-per-agent. Not wrong, but the effective sync rate silently depends on the number of agents; consider counting per-agent or documenting the intent.

### 14. Agent speed/stamina defaults hardcoded in core
- `src/multi_agent_package/core/agent.py:64-72`. `predator=1, prey=3, stamina=10` are baked into the core `Agent`. Config overrides them via `build_agents`, but experiment parameters living in core is the "config drives everything" smell the project warns against — and a prey default *faster* than the predator is a surprising asymmetry to hardcode.

### 15. Duplicate action-decoding source of truth
- `src/multi_agent_package/core/agent.py:76` builds `_actions_to_directions`, used as a fallback in `gridworld.py:254` when `action_space_plugin is None`. Two independent action→direction maps (agent's vs the action plugin) can drift. Prefer a single source (the plugin).

### 16. `scripts/test_baseline.py` pytest-collection collision
- `src/multi_agent_package/scripts/test_baseline.py`. Named `test_*.py`, so a bare `pytest` (unscoped) tries to collect `test_algorithm(algo_name)` and errors on the missing `algo_name` fixture. CI scopes to `tests/` so it's safe there, but rename it (e.g. `sanity_check_baselines.py`) to avoid the trap. Minor: line 37 comment says "no obstacles" but `perc_num_obstacle=10`; `config["cql_alpha"]` (line 57) is read by no algorithm.

### 17. `sweep.py` hardcodes the swept parameter
- `src/multi_agent_package/scripts/sweep.py:30`: `sweep("radius", [1,2,3,4])`. The sweep target/values should come from CLI args or config, not be edited in code.

---

## INFO / cosmetic

### 18. Coordinate/visual notes
- Action "Up" (`1 → [0,1]`) increases `y`, which renders **downward** in Pygame's y-down frame, so compass labels are visually inverted (consistent, and `cross_actions.py` documents the convention). Also `base_reward`'s `-200` obstacle penalty (`gridworld.py:210,228`) is only reachable when `block_agents_by_obstacles=False`, since movement onto obstacles is otherwise blocked — a mostly-dead branch.

---

## Suggested remediation order

1. **#2 `local_radius` slot instability** and **#1 `allow_cell_sharing` no-op** — both are silent correctness bugs that affect results; #2 already has an open issue (#34).
2. **#3 truncation bootstrap** — affects the validity of every learning curve; low-effort, high-value fix.
3. **#6 `sweep.py`**, **#5 `evaluate.py`**, **#4 `evaluate()`**, **#7 `render.py`**, **#8 `render_mode`** — tooling that is broken or misleading for users.
4. The LOW items as cleanup, ideally alongside the files they touch.
