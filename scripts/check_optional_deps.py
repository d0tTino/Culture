#!/usr/bin/env python3
"""Verify presence of optional dependencies for Culture.ai.

This script checks whether commonly used optional packages are installed and
prints installation hints if any are missing. These packages enable certain
tests and features but are not strictly required to run the project.
"""

from __future__ import annotations

import importlib.util

OPTIONAL_DEPS = {
    "chromadb": "pip install chromadb",
    "fastapi": "pip install fastapi",
    "asyncpg": "pip install asyncpg",
}


def have_module(name: str) -> bool:
    """Return ``True`` if the given module can be imported."""
    return importlib.util.find_spec(name) is not None


def main() -> int:
    missing = [pkg for pkg in OPTIONAL_DEPS if not have_module(pkg)]
    if not missing:
        print("All optional dependencies are installed.")
        return 0

    print("Missing optional packages detected:\n")
    for pkg in missing:
        hint = OPTIONAL_DEPS[pkg]
        print(f"  - {pkg} -> install with: {hint}")

    print("\nInstall them to enable features that rely on these packages.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
