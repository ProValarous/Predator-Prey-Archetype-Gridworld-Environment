"""
Suggest the next semantic version based on conventional-commit messages
since the last git tag.

Classification (highest match wins):
  - "BREAKING CHANGE" in a commit body, or a "!" after the type
    (e.g. "feat!:", "fix!:")                      -> MAJOR
  - "feat:" / "feat(scope):"                       -> MINOR
  - anything else (fix, chore, docs, style, test,
    refactor, ci, perf, ...), or no tag yet        -> PATCH

This only computes the release-segment bump (X.Y.Z). Whether the result
should carry a pre-release suffix (-beta, -rc1, ...) or ship final is a
human decision, not something this script guesses.

Usage:
    python scripts/suggest_next_version.py [--base-ref <tag_or_commit>]

Exits non-zero only on a real error (e.g. not a git repo); an empty
commit range or an unparsable current version are reported, not fatal.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

BREAKING_RE = re.compile(r"^\w+(\([^)]*\))?!:|BREAKING[ -]CHANGE", re.MULTILINE)
FEAT_RE = re.compile(r"^feat(\([^)]*\))?:", re.MULTILINE)
VERSION_RE = re.compile(r'version\s*=\s*"([^"]+)"')
RELEASE_SEGMENT_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def run(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def latest_tag() -> str | None:
    try:
        return run("describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        return None


def commits_since(base_ref: str | None) -> list[str]:
    range_spec = f"{base_ref}..HEAD" if base_ref else "HEAD"
    log = run("log", range_spec, "--pretty=format:%B%x00")
    if not log:
        return []
    return [c for c in log.split("\x00") if c.strip()]


def classify(commit_messages: list[str]) -> str:
    """Return 'major', 'minor', or 'patch' for the whole commit range."""
    joined = "\n".join(commit_messages)
    if BREAKING_RE.search(joined):
        return "major"
    if FEAT_RE.search(joined):
        return "minor"
    return "patch"


def current_release_segment() -> tuple[int, int, int]:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise SystemExit('Could not find version = "..." in pyproject.toml')
    seg_match = RELEASE_SEGMENT_RE.match(match.group(1))
    if not seg_match:
        raise SystemExit(f"Could not parse release segment from {match.group(1)!r}")
    return tuple(int(part) for part in seg_match.groups())  # type: ignore[return-value]


def bump(current: tuple[int, int, int], level: str) -> tuple[int, int, int]:
    major, minor, patch = current
    if level == "major":
        return (major + 1, 0, 0)
    if level == "minor":
        return (major, minor + 1, 0)
    return (major, minor, patch + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Compare against this ref instead of the latest tag",
    )
    args = parser.parse_args()

    base_ref = args.base_ref or latest_tag()
    commits = commits_since(base_ref)
    level = classify(commits) if commits else "patch"
    current = current_release_segment()
    suggested = bump(current, level)

    current_str = ".".join(map(str, current))
    suggested_str = ".".join(map(str, suggested))

    print(f"Base ref:          {base_ref or '(none, no tags yet)'}")
    print(f"Commits examined:  {len(commits)}")
    print(f"Bump level:        {level}")
    print(f"Current version:   {current_str}")
    print(f"Suggested version: {suggested_str}")
    print(
        "\nThis is the release-segment suggestion only. Whether it ships as "
        "a final release or a -beta/-rc pre-release is your call."
    )

    # Machine-readable output for CI consumers.
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(f"bump_level={level}\n")
            fh.write(f"current_version={current_str}\n")
            fh.write(f"suggested_version={suggested_str}\n")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"git command failed: {exc}", file=sys.stderr)
        sys.exit(1)
