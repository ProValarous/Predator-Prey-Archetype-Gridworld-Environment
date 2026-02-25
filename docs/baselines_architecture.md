# Baseline Learning Package Architecture (IQL + CQL)

## Purpose and Scope

This document defines an **opinionated architecture** for a baseline learning package that consumes (but never modifies) `multi_agent_package`.

### Non-negotiable boundaries

- The environment core remains immutable.
- The baseline package is an optional consumer and can be installed/uninstalled independently.
- Importing environment modules must not import torch or training dependencies.
- Algorithm selection (IQL vs CQL) is configuration-driven (YAML), not code-driven.

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
- Makes adding future algorithms (e.g., SAC, QMIX variants) additive rather than refactor-heavy.

---

## Interface Design: IQL/CQL ↔ Environment

## 1) Adapter boundary (required)

Both IQL and CQL should only interact with the environment via a **small adapter protocol**:

- `reset(seed) -> ObsDict`
- `step(ActionDict) -> (ObsDict, RewardDict, DoneDict, InfoDict)`
- `get_agent_ids() -> list[str]`
- `get_action_space(agent_id)`
- `get_observation_space(agent_id)`

No algorithm code should import `multi_agent_package.core.*` directly.

## 2) Observation/reward usage through environment configuration

The baseline package should not construct concrete observation/reward classes itself.
Instead, it should:

1. Load environment YAML.
2. Pass observation/reward names to environment script/factory path already backed by registry.
3. Receive resolved observations/rewards from environment outputs.

This ensures algorithm code remains agnostic to whether observation is `default`, `local_radius`, or future plugins.

## 3) Single-agent and multi-agent compatibility

Implement a shared `AgentBatchView` abstraction:

- **Single-agent mode**: selects one controlled agent and masks others.
- **Multi-agent mode**: shared trainer loop with per-agent buffers or parameter sharing strategy from config.

Avoid separate bespoke training loops for each mode; use one orchestrator with mode flags.

---

## Registry Strategy (Opinionated)

## What to do

Use **two registries**, each in its own package boundary:

1. Environment registries (already existing): observation and reward plugin lookup.
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
3. Build environment through adapter using env config (which resolves registries internally).
4. Instantiate algorithm from baseline algorithm registry.
5. Run train/eval loop.
6. Persist artifacts:
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
  env_config_path: configs/env.yaml
  agents_config_path: configs/agents.yaml
  rewards_config_path: configs/rewards.yaml
  observations_config_path: configs/observations.yaml

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

Algorithm swap is then exactly one line (`baseline.algorithm`).

---

## Reproducibility Guardrails

To keep research-grade determinism, enforce:

- **Strict seed plumbing**: one experiment seed expanded deterministically into env seed + dataloader seed + eval seed.
- **Config immutability at runtime**: dump fully resolved config file next to outputs.
- **Version pinning**: capture package versions and commit SHA in every run metadata.
- **Deterministic evaluation mode**: fixed seeds, fixed episode count, no exploration noise.
- **Registry resolution logging**: log which observation/reward plugin names resolved to which classes.

---

## High-Risk Failure Modes to Avoid

1. **Hidden torch import in environment package**
   - Violates optional dependency boundary.
   - Symptom: `import multi_agent_package` fails without torch.

2. **Direct import of environment internals from algorithms**
   - Couples trainers to core dynamics and breaks immutability expectations.

3. **Dual registry copies for observations/rewards**
   - Creates silent mismatches when one registry updates and the other does not.

4. **Algorithm-dependent environment behavior**
   - Any branching like `if algorithm == "cql": env.foo = ...` is a design smell and should be rejected.

5. **Non-serialized run state**
   - If seed/config/commit are not captured, results are not reproducible.

6. **Implicit defaults in code that override YAML**
   - Leads to “same config, different run” outcomes.

7. **Per-script bespoke launch paths**
   - Multiple entry points with different setup logic silently break parity across experiments.

---

## Explicitly Bad Ideas (Reject These)

- Putting trainer classes under `multi_agent_package/scripts/`.
- Allowing baselines to mutate env objects beyond standard API calls.
- Building environment plugins inside baseline package and bypassing env registries.
- Auto-discovering algorithms via dynamic imports with side effects.
- Hard-coding IQL/CQL branches in runner logic instead of registry-based factories.

---

## Practical Next Steps

1. Keep environment package untouched.
2. Replace current ad hoc baseline scripts with the directory contract above.
3. Introduce `baselines.algorithms.registry` and a single YAML-driven runner.
4. Add reproducibility tests (seed repeatability + config parity).
5. Keep baseline dependency extras isolated (e.g., `pip install .[baselines]`).

This yields a clean research codebase: immutable environment, optional learning package, and reproducible configuration-driven experiments.
