# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0-beta] - 2026-08-06

### Added
- First PyPI release: the package is published as
  [`ppage`](https://pypi.org/project/ppage/) (`pip install ppage`), matching
  the repository and import names. A tag-triggered release workflow
  (`release.yml`) builds the sdist and wheel, publishes to PyPI via trusted
  publishing (OIDC, no stored tokens), and creates a GitHub Release with
  notes extracted from this changelog. A companion `version-check.yml`
  suggests the next version on pull requests.

### Changed
- **Breaking:** the import surface is now a single top-level package, `ppage`,
  in preparation for the first PyPI release. `multi_agent_package` became
  `ppage` and `baselines` became `ppage.baselines`; all imports, module CLIs
  (`python -m ppage.scripts...`, `python -m ppage.baselines...`), docs, and CI
  paths were updated. Reinstall with `pip install -e .` after pulling.
- **Breaking:** `torch` and `tensorboard` are no longer hard dependencies.
  They moved to a new `baselines` extra (`pip install -e ".[baselines]"`),
  so installing the environment alone no longer pulls in PyTorch. The tabular
  baselines (IQL, CQL, MixedTrainer) still work without the extra.
- Repository moved to the `UHUMALAB` GitHub org as `UHUMALAB/PPAGE`; repo,
  docs-site, and issue URLs across README, docs, packaging metadata, and
  `CITATION.cff` now point there.

### Fixed
- `pyproject.toml` authors now credit all six contributors, matching the
  README citation and `CITATION.cff` (#65). The README BibTeX entry uses
  real names instead of GitHub handles.

## [0.8.0-beta] - 2026-07-27

### Added
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
