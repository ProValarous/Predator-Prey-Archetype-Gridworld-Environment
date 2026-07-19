# Comparative Review: PR #22 vs #23 vs #24 (DQN implementations)

> **Historical maintainer note.** A three-way review of early DQN PRs (now merged),
> kept for provenance and intentionally not in the site navigation. For current DQN
> docs see [DQN](algorithms/dqn.md) and [DQN Variants](concepts/dqn-variants.md).


Reviewed against merge-base `df8c828` (`pipeline-sdd`) using local diffs (`git diff <merge-base> <pr-head>`), since PR #22's GitHub diff exceeds the API's 300-file limit (caused by an accidentally committed/removed `.new_venv/` — noise, not real changes).

| | PR #22 | PR #23 | PR #24 |
|---|---|---|---|
| Title | Feat/vanila-dqn-affan | ssd-pipeline PR #1 | DQN Implementation and Diagonal action plugin |
| Author/fork | affan002 | afshadGit | Areesha06 |
| Real diff size (excl. venv noise) | 19 files, +952/-26 | 40 files, +1423/-10 | 29 files, +1706/-19 |
| ML framework | PyTorch | PyTorch | **TensorFlow/Keras** (new dep, on top of existing `torch`) |
| New feature focus | Generalized observation→vector encoding contract | Speed-asymmetric predator/prey sweep | Diagonal ("cross") action space |
| Test methods (`test_baselines_dqn.py`) | 12 | 14 | 45 |

## 1. Observation encoding / state representation

**PR #22 and #24 share a near byte-identical redesign** of `observations/base.py`, `default.py`, `local_only.py`, `local_radius.py`, `relative.py`, `absolute.py` — diffing them shows only whitespace differences. Both introduce the same `build(env)` / `encode(observation, env)` contract, the same `_agent_type_id` / `_team_features` / `_vector` helpers, and wire `env.observation_encoder = observation_builder.encode` identically in `run_from_config.py`. This is either shared authorship or one branch built directly on the other's commits — worth confirming with the two authors before merging both, since merging either first will make the second a no-op diff.

- **PR #22 — best of the three here.** `DQN._encode_observation` calls `env.observation_encoder` generically, so the *same* DQN class works with `default`, `local_only`, `local_radius`, `absolute`, and `relative` observation types purely by changing `configs/observations.yaml`. `local_radius.encode()` pads to `len(env.agents)-1` / `len(env._obstacle_location)` automatically — no manual "max agents" tuning needed. `state_dim` and `action_dim` are both auto-inferred at construction time (`_infer_action_dim`, encoding the first reset observation) with a fail-fast check if `observation_encoder` is missing.
- **PR #24 — inconsistent.** It ships the *identical* generalized encoder architecture (see above) but its own `DQN._encode_state` **never calls `env.observation_encoder`**. Instead it hardcodes a bespoke encoder for the `relative` observation, and only the first entry of `obs["agents"]` (`next(iter(agents_seen.values()))`), i.e. it silently assumes exactly 2 agents (1v1). Point it at `local_radius`/`absolute`/`local_only` or a 2v2 config and it will KeyError or silently read the wrong opponent. The generalized encoder work is present in the repo but effectively dead code from DQN's point of view.
- **PR #23 — pragmatic middle ground.** Doesn't touch `base.py`/`relative.py`/`absolute.py` at all; instead it builds its own encoder directly inside `dqn.py`, hardcoded to the `local_radius` observation's dict shape (`visible_agents`, `visible_obstacles`). Unlike #24, it *does* generalize across N agents via configurable `max_other_agents` / `max_visible_obstacles` (zero-padded slots, obstacles sorted nearest-first) — closer in spirit to #22's flexibility, but only for one observation type, and it extended `local_radius.build()` to add a `speed` field per visible agent without adding a matching `encode()` — so it can't be swapped in under #22's generic-encoder story without extra work.

**Verdict:** #22's encoder abstraction is the most correct and reusable; #23's DQN-side encoder is the most agent-count-robust for the one observation type it targets; #24's encoder is a liability because it silently breaks outside 1v1 despite claiming to support the general contract.

## 2. DQN algorithm core

All three are correctly-structured independent (one-network-per-agent) DQN: replay buffer, target network synced periodically, epsilon-greedy, Bellman target via `torch.no_grad()`/frozen target net.

- **PR #24's replay buffer** (`replay_buffer.py`) is the most efficient design of the three: preallocated numpy ring buffer (`_states`, `_actions`, ... arrays with a wrapping `_write_ptr`), giving O(1) vectorized random sampling regardless of buffer fullness. #22 and #23 both use a `collections.deque` of `Transition` dataclasses and convert to a list before indexing on every `sample()` call — functionally fine at this env's scale, but O(n) per sample as the deque grows, and #24's approach is the one that would actually scale.
- **PR #24 switching to TensorFlow/Keras is the weakest architectural call of the three.** The rest of the codebase's baselines (IQL/CQL, and #22/#23's DQN) all use PyTorch — adding TensorFlow as a second full DL framework for one baseline increases install size, CI time, and maintenance burden (two sets of GPU/CPU wheel concerns, two autodiff mental models for contributors) for no capability #22/#23 don't already have with `torch`.
- **PR #24 has leftover dead code**: `q_network.py` and `replay_buffer.py` both carry a large commented-out first draft of the same class directly above the real implementation (~80 and ~45 lines respectively). Should be deleted before merge, not shipped.
- **PR #23's `_update_agent`** adds `torch.nn.utils.clip_grad_norm_(..., 5.0)` before the optimizer step — gradient clipping that #22 and #24's PyTorch-equivalent path don't have (#24 has `grad_clip` via Keras `clipnorm`, so it's covered there too; #22 is the one with no gradient clipping at all).
- **PR #22's `train()`** has the most useful first-episode/first-step debug trace (`debug_first_episode`) printing shapes/rewards for the very first transition — genuinely helpful when wiring a new observation plugin, and cheap since it's a one-off log line.
- **PR #23's `train()`** is the only one that writes a per-episode CSV of reward/loss (`curves_path`) — useful for the sweep-style experiments it's built around, and the kind of artifact you want when comparing 4 speed-ratio configs against each other.
- **Config validation:** #24 is the most defensive — it validates `action_dim` from config against `env.action_space_plugin.n_actions` and raises a clear `ValueError` on mismatch, and warns if `buffer_size < batch_size`. #22 sidesteps the problem by auto-inferring `action_dim` from the plugin instead of trusting a hand-typed config value (arguably a better UX than "validate after the fact"). #23 neither infers robustly from the plugin nor validates against it — `action_dim` defaults to `env.action_space.n` but a user who sets it wrong in YAML gets no warning, just wrong-shaped output layer.

## 3. Action-space extensions

Independent, unrelated features — no overlap:

- **PR #24 — `CrossActionSpace` (diagonal movement).** Clean, small, well-documented plugin (4 diagonals + NOOP), correctly implemented as a sibling of `DiscreteActionSpace` (not a subclass pretending to be one), with an explicit coordinate-convention note in the docstring to avoid the "NE looks like SE on screen" confusion. This is the cleanest single file across all three PRs.
- **PR #23 — `SpeedDiscreteActionSpace` + `SpeedWrapper` (speed asymmetry).** Lets predator/prey move at different cells-per-step via a transparent env wrapper that replays sub-steps, sums rewards, and deducts stamina, short-circuiting on `terminated`/`truncated` so a faster agent can't "overshoot" a capture. This is the most functionally ambitious piece of the three PRs — it changes environment dynamics, not just the DQN baseline — and it ships with a real experiment: 1:1/2:1/3:1 speed-ratio configs (`dqn_1v1`, `dqn_speed1/2/3`) to actually test the hypothesis. Correspondingly it's the riskiest to review carefully since it touches shared env-stepping semantics that #22/#24 don't touch at all.

## 4. Testing

- **PR #24 has by far the deepest test suite** (45 test methods vs. 14 and 12): it separately unit-tests `QNetwork` (input validation, shape checks) and `ReplayBuffer` (ring-buffer wraparound, sampling) in isolation, in addition to the `DQN` class itself — the other two PRs only test the `DQN` class as a black box.
- #22 and #23 are comparable in depth (12–14 tests) and both cover init/action-selection/train/persistence; #23 adds a dedicated `TestDQNEncoding` class exercising its bespoke `local_radius` vector layout directly.
- None of the three were executed live in this review (would require installing/reconciling `torch` + `tensorflow` in one environment); this assessment is structural (what's tested), not a confirmation that all tests currently pass.

## 5. Documentation & config hygiene

- PR #22 adds `docs/observation-encoding-summary.md` explaining the new `build`/`encode` contract — the only PR with a standalone design doc.
- PR #22's shipped `configs/experiment_dqn.yaml` defaults to `episodes: 5` — looks like a leftover smoke-test value rather than a usable training default; #23 (200) and #24 (1000) ship sensible defaults.
- PR #23 ships the most complete config surface for its feature: 4 full config sets (`dqn_1v1`, `dqn_speed1/2/3`) each with actions/agents/env/experiment/observations/rewards YAML, i.e. a reproducible experiment matrix rather than a single config.

## Summary — what's best where

| Concern | Best | Why |
|---|---|---|
| Observation-encoding abstraction | **#22** | Generic `env.observation_encoder`, works across all 5 observation plugins without touching DQN code |
| Replay buffer implementation | **#24** | Preallocated numpy ring buffer, O(1) vectorized sampling |
| Action-space extension code quality | **#24** (`CrossActionSpace`) | Small, correct, clearly documented sibling plugin |
| New experimental capability | **#23** (`SpeedWrapper` + sweep) | Only PR that changes env dynamics and ships a matched multi-config experiment |
| Config/action_dim validation | **#24** | Explicit mismatch checks against the action plugin, not just trust-the-config |
| Test coverage depth | **#24** | Unit tests `QNetwork`/`ReplayBuffer` in isolation, 3x the test count |
| Dependency hygiene | **#22 / #23** (tie) | Stay on the repo's existing `torch`; #24 adds `tensorflow` as a second DL framework |
| Generalizes beyond 1v1 / one obs type | **#22** (fully), **#23** (for `local_radius` only) | #24's DQN hardcodes a 1v1 `relative`-only encoder despite having the general contract available |
| Code cleanliness | **#22 / #23** (tie) | #24 ships ~125 lines of commented-out dead code in `q_network.py`/`replay_buffer.py` |

**None of the three is a strict superset of the others** — they're largely additive (different observation contract vs. speed mechanic vs. diagonal actions) with one real conflict (PR #22 and #24 both redesign the same observation files near-identically) and one real regression risk (PR #24's DQN not actually using the generalized encoder it ships). If merging more than one, land #22 first (generic encoder + docs), then #23 (speed mechanic, keep its local `local_radius`-specific encoder since #22 doesn't cover per-agent speed yet), then #24 — but before merging #24, either (a) rewrite its `_encode_state` to call `env.observation_encoder` like #22 so it isn't silently 1v1-only, or (b) drop the TensorFlow dependency and reuse #22/#23's PyTorch `QNetwork`/`ReplayBuffer`, keeping only its genuinely good pieces: `CrossActionSpace`, the numpy ring-buffer design, and the expanded test suite.
