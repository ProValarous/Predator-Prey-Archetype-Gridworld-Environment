"""
Extract one version's section from CHANGELOG.md, for use as GitHub
release notes.

Usage:
    python scripts/extract_changelog_section.py <version>

<version> is matched against a "## [<version>]" heading (the brackets
and leading "v" on the argument are optional, e.g. both "0.9.0-beta"
and "v0.9.0-beta" find "## [0.9.0-beta] - 2026-07-28"). Prints the
section body (excluding the heading) to stdout. Exits non-zero with a
clear message if the version has no changelog section yet, callers
should fall back to a generic message rather than fail the release.
"""

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: extract_changelog_section.py <version>")

    version = sys.argv[1].lstrip("v")
    text = CHANGELOG.read_text(encoding="utf-8")

    heading_re = re.compile(r"^## \[" + re.escape(version) + r"\].*$", re.MULTILINE)
    match = heading_re.search(text)
    if not match:
        raise SystemExit(f"No CHANGELOG.md section found for version {version!r}")

    start = match.end()
    next_heading = re.search(r"^## \[", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)

    print(text[start:end].strip())


if __name__ == "__main__":
    main()
