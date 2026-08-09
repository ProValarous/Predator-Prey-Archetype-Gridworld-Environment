# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New baseline: **JAL-GT** (Joint-Action Learning with Game Theory —
  Correlated Q-learning, Algorithm 7 in Albrecht, Christianos & Tuyls 2024,
  Section 6.2). `src/baselines/JALGT/jal_gt.py`: one joint-action value table
  per agent, solved as a correlated-equilibrium linear program
  (`scipy.optimize.linprog`) at every visited state rather than a greedy
  argmax. Verified directly against the source textbook's own worked
  examples (exact Prisoner's Dilemma result, Chicken-game welfare bound).
  After root-causing and fixing an initial learning-effectiveness gap vs.
  CQL (prey reward shaping, random Q-init, longer episodes — see
  `docs/algorithms/jal-gt.md`), JAL-GT shows a modest, reproducible
  capture-rate edge over CQL (~+1.76pp average) on a harder, real-prey-evasion
  task, confirmed via paired runs across a complete, symmetric set of 6
  independent environment layouts (5 of 6 favor JAL-GT),
  at a real compute cost (~45-70x slower than CQL). New `prey_distance`
  reward (`src/multi_agent_package/rewards/prey_distance.py`) added
  alongside it.
- Documentation: new "Scope and Generalization Roadmap" page
  (`docs/overview/scope-and-roadmap.md`) giving an honest assessment of what the
  environment can express today (spatial pursuit with mixed cooperation and
  competition) versus the broader vision, the core boundaries with file
  references, and a two-tier generalization roadmap (a registered wrapper layer,
  then a core v2). Also serves as scoping material for a JOSS submission.
- CI: Python 3.10 / 3.11 / 3.12 test matrix, an editable-install import smoke
  test, a coverage gate (`--cov-fail-under=75`, currently ~81%), a
  `mkdocs build --strict` docs job, and a pre-commit lint job (#46).
- Tooling: `.pre-commit-config.yaml`, Dependabot config, and a weekly `pip-audit`
  security workflow (#46).
- Tests: a determinism/reproducibility guard (`tests/test_determinism.py`) that
  asserts a fixed seed and config reproduce an identical reset layout and
  trajectory (#46).

### Changed
- Repository was renamed on GitHub to
  `ProValarous/PPAGE-Predator-Prey-Archetype-Gridworld-Environment`; updated all
  repo URLs in `README.md`, `pyproject.toml`, `mkdocs.yml`, and the docs to the
  new location.
- Dependabot now groups updates into a single pull request per ecosystem (pip and
  github-actions) instead of one PR per dependency, to cut branch and PR noise.

### Removed
- Repository hygiene: untracked build and environment artifacts that had been
  committed before `.gitignore` covered them, and extended `.gitignore` to keep
  them out. Files remain on disk; only version-control entries were removed:
  - the entire `.new_venv/` virtualenv (481 files),
  - both `*.egg-info/` packaging-metadata directories,
  - stale `__pycache__/*.pyc` caches,
  - a 45MB `wiki_compressed.zip` archive,
  - a stray `test_file.md`,
  - ~46MB of IQL TensorBoard logs and trained `.npz` artifacts,
  - a duplicate copy of `PPAGE_overview.png`,
  - generated slide decks (`slides/*.pdf`, `slides/*.pptx`).

### Notes
- These removals affect the branch tip only; the large blobs still exist in git
  history, so clone size is unchanged. A history rewrite (`git filter-repo`) to
  purge them is documented as an optional follow-up.
