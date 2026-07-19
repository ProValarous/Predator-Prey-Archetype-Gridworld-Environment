# DQN Integration Notes — merging PR #22, #23, #24

> **Historical maintainer note.** This records how the DQN baseline was assembled
> from three early PRs (now merged). It is kept for provenance and is intentionally
> not in the site navigation. For current DQN docs see [DQN](algorithms/dqn.md) and
> [DQN Variants](concepts/dqn-variants.md).


Working notes for assembling one DQN baseline out of the three open PRs. Source PRs reviewed at their current heads (`pr22-head` = affan002:feat/vanila-dqn-affan, `pr23-head` = afshadGit:ssd-pipeline, `pr24-head` = Areesha06:pipeline-sdd), diffed against merge-base `df8c828`. See [pr-comparison-22-23-24.md](pr-comparison-22-23-24.md) for the full three-way review this builds on.

## Component checklist

| Component | Take from | Files | Notes |
|---|---|---|---|
| Observation plugins (`build`) | **#22** | `observations/{base,default,local_only,local_radius,relative,absolute}.py` | #22 and #24 ship a near-identical rewrite of these files (whitespace-only diff) — pick #22's copy as canonical, treat #24's as a duplicate to drop, not a merge. |
| `encode()` on each plugin | **#22** | same files as above | This is the `build()`/`encode()` split — `encode(observation, env)` returns the fixed-length numeric vector DQN consumes. Wire it as `env.observation_encoder = observation_builder.encode` in `run_from_config.py`, same line #22 and #24 already both added. |
| Cross (diagonal) action plugin | **#24** | `actions/cross_actions.py` | Clean, self-contained sibling of `DiscreteActionSpace` (not a subclass) — 4 diagonals + NOOP, `n_actions = 5`. No changes needed to adopt as-is. |
| Speed action space + wrapper | **#23** | `actions/speed_discrete.py`, `wrappers/speed.py` | `SpeedDiscreteActionSpace.to_moves()` + `SpeedWrapper` replay sub-steps, sum rewards, deduct stamina, short-circuit on `terminated`/`truncated`. `SpeedWrapper.__getattr__` proxies unknown attributes to the inner env — confirm `env.observation_encoder` / `env.action_space_plugin` are attached to the **inner** env *before* wrapping, so the proxy exposes them (see integration-order caveat below). |
| User-facing config + scripts | **#23** | `scripts/run_dqn.py`, `configs/dqn_1v1/*`, `configs/dqn_speed{1,2,3}/*` | #23's `run_dqn.py` mirrors `run_iql.py`'s existing CLI shape (`--mode train\|eval`, `--config-dir`, `--save-path`/`--load-path`, `--render`) and is fully YAML-driven. This is the one to standardize on — #22's equivalent (`run_dqn_testbed.py`) is explicitly self-described as a "hardcoded test bed" that bypasses `run_from_config` entirely, and #24's `run_dqn.py` is a near-duplicate of #23's with a stray floating docstring/ASCII-diagram left after `main()`. #23's per-experiment config subfolders (`dqn_1v1/`, `dqn_speed1/2/3/`) are also the only reproducible multi-condition experiment layout of the three — worth keeping as the convention for future sweeps. |
| Replay buffer | **#24** | `DQN/replay_buffer.py` | Preallocated numpy ring buffer (fixed `(capacity, state_dim)` arrays, wrapping write pointer) — O(1) vectorized sampling regardless of fill level, vs. #22/#23's `deque`-of-dataclass + list-cast-per-sample. Requires `state_dim` known at construction time, which is already true in all three `__init__`s (encode the first reset observation up front). **Call-site change required**: #24's buffer API is `push(state, action, reward, next_state, done)` / `sample(batch_size) -> (states, actions, rewards, next_states, dones)` as raw arrays — different from #22/#23's `Transition` dataclass + list-of-objects convention. Drop the `Transition` dataclass entirely when adopting this; rewrite the training-loop transition handling and `_optimize_agent`/`_update_agent` to consume array tuples directly (#24's own `_learn()` already shows this pattern). |
| Config / `action_dim` validation | **#24** | `DQN.__init__` | Validates `action_dim` from config against `env.action_space_plugin.n_actions` (raises `ValueError` on mismatch) and warns if `buffer_size < batch_size`. See recommendation below on how this should combine with #22's auto-inference rather than replace it. |
| Progress report | **#23** | — | **Not actually retrievable.** PR #23's `.gitignore` diff explicitly adds `PROGRESS_REPORT.md` and `PRESENTATION_SCRIPT.txt` as ignored — the file exists in the author's local working tree but was never committed to the branch, so it isn't in the PR diff or any commit reachable from `pr23-head`. Ask afshadGit to attach/paste it directly (PR description, a comment, or a follow-up commit with the ignore rule removed) — it cannot be pulled from git history. |

## DQN algorithm core — which implementation to build on

None of the three `dqn.py` files should be adopted wholesale; the right base is **#22's class skeleton**, with **#24's replay buffer and validation** grafted in and **TensorFlow dropped in favor of staying on PyTorch**. Reasoning below.

### Architectural precedent: check against `BaseAlgorithm` and `IQL`

`src/baselines/base.py` states the contract plainly: *"Algorithms treat env as a black box."* `IQL` (`src/baselines/IQL/iql.py`) follows this literally — it encodes whatever observation dict it receives into a hashable tuple **internally**, with no assumption about structure and no external attribute the environment must be pre-wired with. That's the existing convention in this codebase.

All three DQN PRs *technically* deviate from that in the same direction — they all need a fixed-length numeric vector, which a hashable-tuple scheme doesn't give them — but they diverge in how:

- **#22** attaches the encoding responsibility to the environment (`env.observation_encoder`, wired externally in `run_from_config.py`) and fails fast if it's missing. This makes DQN generic across every observation plugin (`default`, `local_only`, `local_radius`, `absolute`, `relative`) with zero DQN-side code per plugin — the strongest modularity story of the three — at the cost of DQN no longer being usable against a bare `GridWorldEnv` the way `IQL` is; it now requires whatever constructs the env (a script or a test) to also attach `observation_encoder`.
- **#23** and **#24** each embed their own bespoke `_encode_observation`/`_encode_state` directly inside the `DQN` class — closer to `IQL`'s "self-contained, no external wiring" spirit, but at the cost of being hardcoded to one observation shape each (#23: `local_radius`'s `visible_agents`/`visible_obstacles` dict layout; #24: `relative`'s 1v1 `agents` dict, via `next(iter(...))`, which breaks for anything but exactly 2 agents).

Since the observation-plugin checklist above already locks in **#22's `build`/`encode` contract**, the DQN core has to consume it through `env.observation_encoder` — that makes **#22's `dqn.py` the only one of the three actually designed around the encoder contract we're keeping**, so it's the correct base, not a stylistic preference. The `IQL`-style "no external wiring" property is worth a one-line callout in the class docstring as a known, deliberate deviation, but isn't worth abandoning the generic-encoder design over — the alternative is re-hardcoding DQN to one observation shape, which is strictly worse now that five plugins exist.

### Framework: stay on PyTorch, do not carry over TensorFlow from #24

#24 is the only PR that introduces TensorFlow/Keras (`q_network.py`), on top of `torch` which `IQL`/`CQL`/#22/#23 already depend on. Adopting it would mean the repo carries two full deep-learning frameworks for one baseline family, doubling install size and CI time and forcing contributors to context-switch between `nn.Module`/autograd and `keras.Sequential`/`GradientTape` conventions depending on which baseline they're touching. #22's and #23's `QNetwork` (`nn.Module`, configurable hidden layers, linear output head) are functionally equivalent to #24's Keras version and should be used instead — #23's version is marginally preferable since its `hidden_layers` list is fully variable-depth (`for hidden_dim in hidden_layers: ...`) rather than #22's fixed two-hidden-layer shape, and it doesn't touch a second framework.

Also drop #24's ~125 lines of commented-out first-draft code left in `q_network.py`/`replay_buffer.py` — don't carry dead code into the merged version regardless of which parts of #24 are kept.

### `action_dim` handling: combine #22's inference with #24's validation

Don't pick one over the other — they solve different failure modes:

- #22 auto-infers `action_dim` from `env.action_space_plugin.n_actions` when not given, so a user swapping in `CrossActionSpace` (5 actions) or `SpeedDiscreteActionSpace` (5 actions, inherited from `DiscreteActionSpace`) never has to touch YAML.
- #24 validates an *explicitly configured* `action_dim` against the plugin and raises a clear error on mismatch, catching stale YAML after an action-space change.

Merged behavior: if `action_dim` is present in config, validate it against `env.action_space_plugin.n_actions` (#24's check); if absent, infer it from the plugin (#22's behavior) instead of defaulting to a bare number. This keeps zero-config usage working while still catching drift when a value is explicitly pinned. Carry over #24's `buffer_size < batch_size` warning unconditionally — it's a cheap, useful guard with no downside.

### Other pieces worth carrying regardless of "core" choice

- **#22**: `debug_first_episode` / `verbose` first-transition trace — cheap, and genuinely useful when wiring a new observation or action plugin into DQN for the first time (exactly the situation this merge creates three times over).
- **#23**: `torch.nn.utils.clip_grad_norm_(..., 5.0)` before the optimizer step — free numerical-stability improvement once we're committed to PyTorch; neither #22 nor #24's PyTorch path has an equivalent (#24 gets clipping via Keras `clipnorm`, which goes away with TF).
- **#23**: per-episode CSV curve logging (`curves_path`) — pairs naturally with the adopted `dqn_speed1/2/3` sweep configs; without it there's no artifact to compare the three speed ratios against each other.

### Integration-order caveat (`SpeedWrapper` + `observation_encoder`)

`SpeedWrapper.__getattr__` forwards any attribute `SpeedWrapper` doesn't define itself to the wrapped env, so `env.observation_encoder` and `env.action_space_plugin` remain reachable through the wrapper *only if they were attached to the inner `GridWorldEnv` before wrapping*. `build_environment()` in `run_from_config.py` must attach `observation_builder`/`observation_encoder`/`action_space_plugin`/`reward_fn` first, then apply `SpeedWrapper` last. Worth a regression test once #23's wrapper and #22's encoder wiring land in the same `run_from_config.py` — this combination hasn't been exercised together in any of the three PRs individually.

### Persistence (`save`/`load`): keep it simple unless resumability is a requirement

#22's `save`/`load` persists policy/target `state_dict`s only, matching `IQL.save`/`load`'s "persist the learned parameters, nothing else" simplicity. #23's version additionally persists optimizer state and the full replay buffer contents, which is heavier but enables genuinely resuming interrupted training (Adam momentum, in-flight experience). Default to #22's simpler version for consistency with `IQL`; revisit if resumable training across process restarts becomes an actual requirement rather than a nice-to-have.

## Net result

| Piece | Final source |
|---|---|
| Class skeleton, init flow, `observation_encoder` fail-fast check, debug logging | #22 |
| `QNetwork` (PyTorch `nn.Module`, variable hidden layers) | #23 |
| `ReplayBuffer` (numpy ring buffer) | #24, with call sites rewritten for #22's skeleton |
| Gradient clipping | #23 |
| `action_dim` inference + validation | #22 (infer) + #24 (validate-if-configured) |
| `buffer_size`/`batch_size` warning | #24 |
| Save/load | #22 |
| Action plugins | #24 (`CrossActionSpace`) + #23 (`SpeedDiscreteActionSpace`, `SpeedWrapper`) |
| CLI script + config layout | #23 |
| Progress report | Blocked — request from #23's author, not retrievable from git |
