# Git Workflow

This page is the single source of truth for how git is actually used in this
repository: which branch to work from, what CI checks run and why, how to
open a PR, and how to get out of the two situations that come up most —
merge conflicts and a failed `core-guard` check.

If you only read one section, read **[The Golden Rule](#2-the-golden-rule-core-is-off-limits)**
(`core/` is off-limits) and **[step 5, Push and open the PR](#5-push-and-open-the-pr)**.

---

## Branch model

```
STRP    ●───●───●───●───●───●───●───●───●   ← active development (trunk)
             \               \
main    ──────●───────────────●              ← periodic release snapshots
```

| Branch | Role | Who commits here |
| --- | --- | --- |
| **`STRP`** | The active trunk. All feature/fix branches are cut from `STRP`, and all PRs target `STRP`. CI (lint + tests + `core-guard`) runs on every push and PR here. | Everyone, via PR |
| **`main`** | A periodic release snapshot, updated by merging `STRP` in when a maintainer decides a checkpoint is ready. It lags `STRP` between releases — that's expected, not a bug. | Maintainers only, via merge from `STRP` |

**Practical takeaway: branch from `STRP`, and open your PR against `STRP`.**
Don't branch from or target `main` — it won't have the latest plugins,
CI config, or fixes, and your PR will likely show a conflict-ridden diff
against stale code.

### Branches you can ignore

If you run `git branch -a` you'll see a few branches that aren't part of the
active workflow:

- `legacy-ppage` — a frozen snapshot of an old branch, kept for history.
- `pr21-head`, `pr22-head`, `pr23-head`, `pr24-head` — per-PR reference
  snapshots from already-merged, already-closed PRs.

None of these are meant to be branched from or merged into. They're archival.

---

## Setup

```bash
git clone https://github.com/ProValarous/Predator-Prey-Archetype-Gridworld-Environment.git
cd Predator-Prey-Archetype-Gridworld-Environment
git checkout STRP

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
```

> **`pip install -e .` does not work** — the build backend isn't wired up, so
> an editable install won't make `multi_agent_package` importable even though
> `pip` may report success. Every command below uses `PYTHONPATH=src`
> instead, which is the only verified-working setup. See the
> [Quickstart](../README.md#-quickstart) for details.

---

## Making a change

### 1. Branch from `STRP`

```bash
git checkout STRP
git pull origin STRP
git checkout -b fix/short-desc      # or: feat/short-desc
```

Use `fix/...` for bug fixes, `feat/...` for new functionality. Keep the
branch scoped to one change — small, reviewable PRs merge faster.

### 2. The Golden Rule: `core/` is off-limits

> **You may not modify `src/multi_agent_package/core/` (`gridworld.py`,
> `agent.py`).** Extend the system through `observations/`, `rewards/`,
> `actions/`, `wrappers/`, and `registry/` instead.

This isn't just a guideline — it's enforced automatically. The `core-guard`
CI job diffs your PR against `STRP` and **fails the check if any file under
`core/` changed**, regardless of how small the change looks. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for the full rationale and the layered
architecture this protects.

If you find yourself needing to touch `core/`, that's a signal to open an
issue and discuss it with a maintainer first — don't route around the
registries to avoid the check.

### 3. Before you push: run what CI runs

Run the exact same checks CI will run, locally, before pushing:

```bash
black --check .
flake8 .
PYTHONPATH=src pylint src
PYTHONPATH=src python -m pytest tests/ -q
```

If `black --check` fails, just run `black .` (no `--check`) to auto-fix
formatting, then re-stage.

### 4. Commit

There's no strictly enforced commit message format, but the convention
that's actually followed in this repo's history is:

```
<optional type>: <short imperative summary>

<optional body explaining WHY, not what>
```

Common types seen in `git log`: `fix:`, `feat:`/`feature:`, `docs:`,
`refactor:`, `chore:`. When a type doesn't add clarity, a plain imperative
summary (`Add pytest job to CI`) is fine too. Prefer a few focused commits
over one giant one, but don't over-fragment either.

### 5. Push and open the PR

```bash
git push -u origin fix/short-desc
```

Open the PR **against `STRP`** (GitHub usually defaults to this correctly
since you branched from `STRP`, but double-check the base branch before
submitting — it's easy to accidentally target `main`).

---

## What CI actually checks

Defined in [`.github/workflows/ci.yaml`](../.github/workflows/ci.yaml),
triggered on every push and PR to `main`, `master`, or `STRP`:

| Job | Runs on | What it does |
| --- | --- | --- |
| `lint` | push + PR | `black --check .`, `flake8 .`, `PYTHONPATH=src pylint src`. Fails on any formatting or lint violation. `core/`, `miscellenous/`, and `slides/` are excluded from linting by design (see comments in `.flake8` / `.pylintrc`). |
| `test` | push + PR | `PYTHONPATH=src pytest tests/ -q` — the full test suite (registries, plugin contracts, end-to-end training, architecture rules). |
| `core-guard` | **PR only** | Diffs the PR's base and head SHAs; fails if any file under `src/multi_agent_package/core/` was touched. Never runs on plain pushes (there's no "PR diff" to check). |

All three must pass before merging. There is currently no branch protection
rule enforcing this at the GitHub level — reviewers should confirm the
checks are green before approving.

---

## Handling merge conflicts

The most common source of conflicts in this repo isn't competing logic —
it's **Black reformatting touching the same lines as your change**. If you
get a conflict:

1. Read both sides. If one side is purely whitespace/line-wrapping and the
   other has your actual logic change, keep the logic side and discard the
   formatting-only side.
2. After resolving, run `black .` on the resolved file(s) — the manual
   resolution often isn't Black-formatted itself.
3. Re-run the full check list from [step 3 above](#3-before-you-push-run-what-ci-runs)
   before pushing the resolution.

To sync your branch with the latest `STRP` mid-PR:

```bash
git fetch origin
git merge origin/STRP        # or: git rebase origin/STRP
# resolve conflicts, then:
black .
PYTHONPATH=src python -m pytest tests/ -q
git push
```

**If your PR lives on someone else's fork and you're not that contributor**,
resolving and pushing the fix on their behalf requires their awareness first
— even with "allow edits by maintainers" enabled, it's their branch. Comment
on the PR explaining the conflict and what needs to change instead of
pushing to it unannounced.

---

## Merging

GitHub's squash, merge-commit, and rebase-merge options are all enabled on
this repo — there's no server-side restriction to one method, and history
so far is a mix of both squash and merge commits. Squash is the better
default for a small, single-purpose PR (one clean commit on `STRP`); use a
merge commit if the PR's individual commits are independently meaningful
and worth preserving. Branches are **not** auto-deleted on merge — delete
yours manually after merging (`git push origin --delete fix/short-desc`,
or the "Delete branch" button on the merged PR page).

---

## Quick reference

```bash
# One-time setup
git clone https://github.com/ProValarous/Predator-Prey-Archetype-Gridworld-Environment.git
cd Predator-Prey-Archetype-Gridworld-Environment
pip install -r requirements-dev.txt

# Start work
git checkout STRP && git pull origin STRP
git checkout -b fix/my-fix

# ... make changes ...

# Verify before pushing
black --check . && flake8 . && PYTHONPATH=src pylint src && PYTHONPATH=src python -m pytest tests/ -q

git add <files>
git commit -m "fix: short description"
git push -u origin fix/my-fix
# open PR against STRP
```
