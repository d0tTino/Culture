#!/usr/bin/env python3
"""Check Markdown files for broken relative links."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LINK_RE = re.compile(r"\[(?:[^\]]+)\]\(([^)]+)\)")
EXCLUDE_DIRS = {".git", "node_modules", "dspy_ai-2.6.27.dist-info"}


def is_relative(link: str) -> bool:
    """Return True if the link is relative."""
    if link.startswith(("http://", "https://", "mailto:", "#", "/")):
        return False
    return True


def check_file(md_file: Path) -> list[str]:
    broken: list[str] = []
    try:
        text = md_file.read_text(encoding="utf-8")
    except Exception as e:  # pragma: no cover - reading errors
        print(f"Failed to read {md_file}: {e}", file=sys.stderr)
        return broken

    for match in LINK_RE.finditer(text):
        link = match.group(1).split("#", 1)[0].split("?", 1)[0]
        if not link or not is_relative(link):
            continue
        target = (md_file.parent / link).resolve()
        if not target.exists():
            broken.append(f"{md_file}:{link}")
    return broken


def main() -> int:
    broken_links: list[str] = []
    for md_file in ROOT.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in md_file.parts):
            continue
        broken_links.extend(check_file(md_file))

    if broken_links:
        print("Broken markdown links detected:", file=sys.stderr)
        for item in broken_links:
            print(f"  {item}", file=sys.stderr)
        return 1
    print("No broken markdown links found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
