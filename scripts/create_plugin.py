#!/usr/bin/env python3
"""Scaffold a minimal Culture plug-in package."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

TEMPLATE_INIT = textwrap.dedent(
    '''
    """Minimal plug-in for Culture."""

    from typing import Any

    from src.extensions import PluginResult, register_agent_behavior


    def log_turn(agent: Any, output: dict[str, Any]) -> None:
        """Log the agent's output."""
        print(f"[{package}] {output}")


    def setup() -> PluginResult:
        """Entry point used by ``load_plugins``."""
        register_agent_behavior(log_turn)
        return {
            "name": "ExampleWidget",
            "script_url": "http://localhost:5173/example.js",
        }
    '''
)

TEMPLATE_PYPROJECT = textwrap.dedent(
    '''
    [build-system]
    requires = ["setuptools>=61.0"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "{package}"
    version = "0.1.0"
    description = "Culture plug-in {package}"
    readme = "README.md"
    requires-python = ">=3.10"

    [project.entry-points."culture.plugins"]
    {package} = "{package}:setup"
    '''
)

README = "# {package}\n\nScaffolded Culture plug-in.\n"


def create_plugin(dest: Path, package: str) -> None:
    """Create plug-in package at *dest* with the given *package* name."""
    (dest / package).mkdir(parents=True, exist_ok=False)
    (dest / package / "__init__.py").write_text(
        TEMPLATE_INIT.replace("{package}", package), encoding="utf-8"
    )
    (dest / "pyproject.toml").write_text(
        TEMPLATE_PYPROJECT.format(package=package), encoding="utf-8"
    )
    (dest / "README.md").write_text(README.format(package=package), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a minimal Culture plug-in")
    parser.add_argument("name", help="Plug-in package name")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory to create the plug-in in (default: current directory)",
    )
    args = parser.parse_args(argv)

    dest = Path(args.path).resolve() / args.name
    if dest.exists():
        print(f"Error: {dest} already exists", file=sys.stderr)
        return 1

    create_plugin(dest, args.name)
    print(f"Created plug-in at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
