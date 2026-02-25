# Baseline Learning Package Architecture (IQL + CQL)

## Purpose and Scope

This document defines an **opinionated architecture** for a baseline learning package that consumes (but never modifies) `multi_agent_package`.

### Non-negotiable boundaries

- The environment core remains immutable.
- The baseline package is an optional consumer and can be installed/uninstalled independently.
- Importing environment modules must not import torch or training dependencies.
- Algorithm selection (IQL vs CQL) is configuration-driven (YAML), not code-driven.

---

## Does this integrate cleanly with the current `multi_agent_package`?

**Yes — if the adapter exactly follows the current API contracts in your codebase.**

### Current environment facts the baseline must respect

- Environment construction is done with `GridWorldEnv(...)` plus `build_agents(...)` patterns already used in `scripts/run_from_config.py`.
- Observation and reward plugins are resolved via existing registries:
  - `get_observation_builder(name, **params)`
  - `get_reward_function(name, weight=..., **params)`
- `env.reset()` returns `(obs, info)`.
- `env.step(actions)` currently returns a **dict** with keys:
  - `obs`, `reward`, `terminated`, `trunc`, `info`

The baseline package must adapt to these interfaces rather than expecting Gymnasium tuple-style step outputs.

---

## Recommended Repository Layout

```text
src/
├── multi_agent_package/                      # immutable environment package
│
└── baselines/                                # optional learning package (consumer)
    ├── __init__.py                           # lightweight, no torch import side-effects
    ├── algorithms/
    │   ├── __init__.py
    │   ├── registry.py                       # algorithm registry ("iql", "cql", ...)
    │   ├── base.py                           # Trainer/Policy protocol contracts
    │   ├── iql/
    │   │   ├── __init__.py
    │   │   ├── config_schema.py
    │   │   ├── trainer.py
    │   │   ├── networks.py
    │   │   └── losses.py
    │   └── cql/
    │       ├── __init__.py
    │       ├── config_schema.py
    │       ├── trainer.py
    │       ├── networks.py
    │       └── losses.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── replay_buffer.py
    │   ├── batch.py
    │   └── episode_store.py
    │
    ├── env_adapter/
    │   ├── __init__.py
    │   ├── interface.py                       # EnvAdapter protocol
    │   ├── gridworld_adapter.py               # wraps multi_agent_package env API
    │   └── spaces.py                          # shape/action-space normalization
    │
    ├── runners/
    │   ├── __init__.py
    │   ├── train.py                           # CLI entry for training
    │   ├── evaluate.py                        # eval-only deterministic runner
    │   └── utils.py                           # seed + logging + checkpoint plumbing
    │
    ├── configs/
    │   ├── experiment/
    │   │   ├── iql_single_agent.yaml
    │   │   ├── iql_multi_agent.yaml
    │   │   ├── cql_single_agent.yaml
    │   │   └── cql_multi_agent.yaml
    │   ├── algorithm/
    │   │   ├── iql.yaml
    │   │   └── cql.yaml
    │   └── schema.yaml
    │
    ├── scripts/
    │   ├── run_baseline.py                    # config-first launcher
    │   └── export_results.py
    │
    └── tests/
        ├── test_registry_wiring.py
        ├── test_seed_reproducibility.py
        └── test_config_parity.py
```

### Why this structure

- Keeps all learning logic under `src/baselines/`.
- Prevents accidental core coupling by forcing access through `env_adapter/`.
- Makes adding future algorithms additive rather than refactor-heavy.

---

## Interface Design: IQL/CQL ↔ Environment

## 1) Adapter boundary (required)

Both IQL and CQL should only interact with the environment via a small adapter protocol:

- `reset(seed) -> (obs_dict, info_dict)`
- `step(action_dict) -> StepBatch`
- `get_agent_ids() -> list[str]`
- `sample_random_action(agent_id) -> int`
- `get_action_n(agent_id) -> int`

Where `StepBatch` is normalized by adapter as:

- `obs: dict[str, object]`
- `reward: dict[str, float]`
- `terminated: bool`
- `truncated: bool`
- `info: dict[str, object]`

This normalization hides the environment’s current dict-style step return so algorithms stay stable if wrapper details evolve.

## 2) Observation/reward usage through environment configuration

The baseline package should not construct concrete observation/reward classes itself.
Instead, it should:

1. Load environment YAML files.
2. Reuse the same registry resolution path (`get_observation_builder`, `get_reward_function`) used by existing scripts.
3. Receive already-resolved observations/rewards from the environment.

This keeps algorithm code agnostic to concrete plugin implementations.

## 3) Single-agent and multi-agent compatibility

Implement a shared `AgentBatchView` abstraction:

- **Single-agent mode**: selects one controlled agent and masks others.
- **Multi-agent mode**: shared trainer loop with per-agent buffers or parameter sharing strategy from config.

Avoid separate bespoke training loops; use one orchestrator with mode flags.

---

## Registry Strategy (Opinionated)

## What to do

Use two registries, each in its own boundary:

1. Environment registries (existing): observation and reward plugin lookup.
2. Baseline algorithm registry (new): maps algorithm key (`"iql"`, `"cql"`) to trainer factory.

## What not to do

- Do **not** mirror environment observation/reward registries inside `baselines`.
- Do **not** import concrete env plugin modules from training code.

Mirroring env registries is a bad idea because it creates drift and duplicate source-of-truth.

---

## Config-Driven Experiment Orchestration

The top-level launcher should read one experiment YAML and orchestrate in this order:

1. Parse and validate config schema.
2. Set global seeds (python, numpy, torch, env).
3. Build agents + environment via adapter.
4. Resolve observation/reward plugins through existing environment registry functions.
5. Instantiate algorithm from baseline algorithm registry.
6. Run train/eval loop.
7. Persist artifacts:
   - resolved config snapshot
   - git commit hash
   - seeds used
   - metrics and checkpoints

### Minimal experiment YAML contract

```yaml
experiment:
  name: iql_predator_prey_v1
  seed: 123
  total_env_steps: 500000

environment:
  config_dir: configs

baseline:
  algorithm: iql    # or cql
  algorithm_config: src/baselines/configs/algorithm/iql.yaml
  mode: multi_agent # or single_agent

runtime:
  device: cpu
  deterministic_torch: true
  num_workers: 1
  log_interval: 1000
  checkpoint_interval: 10000
```

Algorithm swap is one line: `baseline.algorithm`.

---

## Reproducibility Guardrails

To keep research-grade determinism, enforce:

- Strict seed plumbing: one experiment seed expanded deterministically into env seed + dataloader seed + eval seed.
- Config immutability at runtime: dump fully resolved config next to outputs.
- Version capture: package versions and commit SHA in every run metadata.
- Deterministic evaluation mode: fixed seeds, fixed episode count, no exploration noise.
- Registry resolution logging: exact observation/reward plugin names selected per run.

---

## High-Risk Failure Modes to Avoid

1. Hidden torch import in environment package
   - Violates optional dependency boundary.
2. Direct import of `multi_agent_package.core.*` from algorithms
   - Couples trainers to immutable internals.
3. Dual registry copies for observations/rewards
   - Creates silent drift.
4. Algorithm-dependent environment behavior
   - Any `if algorithm == ...` branch that mutates env behavior is a design bug.
5. Non-serialized run state
   - Missing seed/config/commit makes results non-reproducible.
6. Implicit code defaults overriding YAML
   - Produces “same config, different run.”
7. Multiple divergent launch scripts
   - Setup mismatch silently breaks experiment parity.

---

## Explicitly Bad Ideas (Reject These)

- Putting trainers under `multi_agent_package/scripts/`.
- Letting baselines mutate env internals beyond public API calls.
- Re-implementing observation/reward plugin systems inside baselines.
- Auto-discovering algorithms via import side effects.
- Hard-coding IQL/CQL branches in runner logic instead of algorithm registry factories.

---

## Practical Next Steps

1. Keep environment package untouched.
2. Replace ad hoc baseline scripts with this directory contract.
3. Implement `baselines.algorithms.registry` and one YAML-driven runner.
4. Add reproducibility tests (seed repeatability + config parity).
5. Keep baseline dependencies optional (e.g., extras: `.[baselines]`).

This gives a real research-codebase separation: immutable environment, optional learning package, and reproducible config-driven experiments.
