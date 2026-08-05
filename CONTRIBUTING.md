# Contributing

## How to contribute

1. Fork -> branch `fix/short-desc` or `feat/short-desc` off of `STRP` (not `main` — see [Git Workflow](docs/git-workflow.md))
2. Run tests: `pip install -r requirements-dev.txt` && `PYTHONPATH=src python -m pytest tests/ -q`
3. Follow the code style (Black) and add type hints where possible
4. Open a PR against `STRP` and link an issue (if any)

For the full branching model, commit conventions, what CI checks on your PR,
and how to resolve merge conflicts, see **[docs/git-workflow.md](docs/git-workflow.md)**.


# Contributing to Predator–Prey GridWorld

Thank you for your interest in contributing 🎯
This project is designed as a **research and teaching infrastructure** for undergraduate and early-stage researchers working on Multi-Agent Reinforcement Learning (MARL).

To keep the codebase **stable, reproducible, and reviewable**, we enforce strict contribution rules.

Please read this document **before writing any code**.

---

## 🚨 Golden Rule (Read This First)

> **You are not allowed to modify core environment or agent code.**
> You are expected to **extend the system, not edit it**.

Violating this rule will result in your PR being rejected without review.

---

## 📁 Project Philosophy

This repository follows a **layered architecture**:

| Layer           | Purpose                        | Who edits it       |
| --------------- | ------------------------------ | ------------------ |
| `core/`         | Environment & agent primitives | ❌ Maintainers only |
| `observations/` | Observation logic              | ✅ Students         |
| `rewards/`      | Reward logic                   | ✅ Students         |
| `configs/`      | Experiment definitions (YAML)  | ✅ Students         |
| `scripts/`      | Experiment runners             | ⚠️ Limited         |
| `registry/`     | Plugin wiring                  | ⚠️ With care       |

This separation is **intentional** and **non-negotiable**.

---

## ❌ What You Must NOT Do

Do **not**:

* Edit files in:

  * `core/gridworld.py`
  * `core/agent.py`
* Add logic directly inside:

  * `GridWorldEnv.step`
  * `GridWorldEnv.reset`
  * `Agent._get_obs`
* Hard-code parameters in Python
* Add new reward logic inside the environment
* Import rewards or observations directly without registries
* Bypass YAML configuration

These patterns break reproducibility and invalidate experiments.

---

## ✅ What You ARE Allowed to Do

### 1️⃣ Add New Reward Functions

Location:

```
rewards/
```

Steps:

1. Subclass `BaseReward`
2. Use **only**:

   * observations
   * agent identities
   * info dict
3. Register it in `reward_registry.py`
4. Reference it **only via YAML**

✔️ This is Good

❌ Editing `GridWorldEnv.base_reward`

---

### 2️⃣ Add New Observation Schemes

Location:

```
observations/
```

Steps:

1. Subclass `BaseObservation`
2. Assume **partial observability**
3. Never access environment internals directly
4. Register in `observation_registry.py`

✔️ Local radius obs

❌ Reading `env._agents_location`

---

### 3️⃣ Create New Experiments (Preferred Contribution)

Location:

```
configs/
```

You are encouraged to:

* create new YAMLs
* sweep parameters
* run ablations
* reproduce figures

This is the **cleanest and most valuable** contribution.

---

### 4️⃣ Add Scripts (Advanced)

Allowed only if:

* Script is **high-level**
* No environment logic inside
* Uses registries + configs

All scripts must be runnable as:

```bash
python -m ppage.scripts.<script_name>
```

---

## 🧪 Reproducibility Requirements

Every contribution must ensure:

* ✅ Deterministic seeding
* ✅ Config-driven parameters
* ✅ No hidden state
* ✅ No global randomness
* ✅ Same config → same result

If your contribution cannot be reproduced from a YAML file, it will not be accepted.

---

## 🧾 Documentation Expectations

If you add:

* a reward → update `rewards/README.md`
* an observation → update `observations/README.md`
* a script → add docstring + usage example

Code without documentation is considered incomplete.

---

## 🧑‍🎓 For Undergraduate Contributors

This repository is used for:

* course projects
* research papers
* empirical studies

Your code may be:

* reused by others
* evaluated by reviewers
* cited in publications

Write accordingly.

Clean code > clever code.

---

## 🔍 Review Criteria (How Your PR Will Be Judged)

We review contributions based on:

1. Architectural correctness
2. Respect for abstractions
3. Reproducibility
4. Readability
5. Scientific clarity

Performance comes **last**.

---

## 🛑 When in Doubt

If you are unsure:

* where code belongs
* whether you should edit a file
* how to add a feature

**Ask first. Do not guess.**

Open an issue with:

* your goal
* proposed location
* expected behavior

---

## ✅ Final Reminder

This project is infrastructure, not a sandbox.

If everyone edits the core, nobody can trust the results.

Follow the rules.
Extend cleanly.
Document everything.

Happy coding and researching.
