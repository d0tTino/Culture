#!/usr/bin/env python3
"""Run pytest with optional plugin-based adjustments.

This helper detects available plugins like ``pytest-xdist`` and ``pytest-asyncio``
and removes incompatible options from ``pytest.ini`` when they are missing. It
also checks for a few optional runtime libraries (``numpy``, ``sqlalchemy``,
``requests``) and prints a warning when they are not installed so that tests
depending on them can be skipped cleanly.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent.parent
INI_FILE = ROOT / "pytest.ini"


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def strip_xdist_flags(args: list[str]) -> list[str]:
    """Remove ``-n`` options if pytest-xdist is unavailable."""
    cleaned: list[str] = []
    skip = False
    for part in args:
        if skip:
            skip = False
            continue
        if part == "-n":
            skip = True
            continue
        if part.startswith("-n=") or part.startswith("-nauto") or part == "-nauto":
            continue
        cleaned.append(part)
    return cleaned


def main(argv: list[str]) -> int:
    has_xdist = have_module("xdist")
    has_asyncio = have_module("pytest_asyncio")
    # Install dependencies if any are missing unless skipped via env var
    required = [
        "fastapi",
        "pytest_asyncio",
        "sqlalchemy",
        "aiosqlite",
        "zstandard",
        "requests",
        "hypothesis",
        "boto3",
        "moto",
        "numpy",
        "chromadb",
    ]
    skip_install = os.getenv("SKIP_DEP_INSTALL")
    if not skip_install and not all(have_module(mod) for mod in required):
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(ROOT / "requirements.txt"),
                    "-r",
                    str(ROOT / "requirements-dev.txt"),
                ]
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - network issues
            print(
                f"WARNING: dependency installation failed ({exc}). "
                "Continuing with existing packages."
            )

    optional = ["numpy", "sqlalchemy", "requests", "aiosqlite"]
    missing_optional = [mod for mod in optional if not have_module(mod)]

    skip_map = {
        "aiosqlite": [ROOT / "tests" / "integration" / "interfaces" / "test_token_sql.py"],
        "sqlalchemy": [ROOT / "tests" / "integration" / "interfaces" / "test_token_sql.py"],
        "numpy": [ROOT / "tests" / "unit" / "infra" / "test_checkpoint.py"],
        "requests": [
            ROOT / "tests" / "unit" / "utils" / "test_policy.py",
            ROOT / "tests" / "unit" / "infra" / "test_llm_client_url.py",
            ROOT / "tests" / "unit" / "infra" / "test_llm_client_vllm.py",
            ROOT / "tests" / "unit" / "infra" / "test_dspy_ollama_config.py",
            ROOT / "tests" / "unit" / "memory" / "test_role_history.py",
            ROOT / "tests" / "unit" / "interfaces" / "test_discord_policy.py",
            ROOT / "tests" / "integration" / "governance" / "test_law_board_integration.py",
        ],
    }
    ignore_paths: set[Path] = set()
    for mod in missing_optional:
        ignore_paths.update(skip_map.get(mod, []))

    if missing_optional:
        joined = ", ".join(missing_optional)
        msg = f"WARNING: missing optional packages: {joined}. " "Skipping tests that require them."
        print(msg)

    cfg = ConfigParser()
    cfg.read(INI_FILE)
    modified = False

    if "pytest" in cfg:
        addopts = cfg["pytest"].get("addopts", "")
        if not has_xdist and "-n" in addopts:
            parts = addopts.split()
            cleaned: list[str] = []
            skip = False
            for part in parts:
                if skip:
                    skip = False
                    continue
                if part == "-n" or part.startswith("-n"):
                    if part == "-n":
                        skip = True
                    modified = True
                    continue
                cleaned.append(part)
            cfg["pytest"]["addopts"] = " ".join(cleaned)
        if not has_asyncio and cfg["pytest"].get("asyncio_mode"):
            cfg["pytest"].pop("asyncio_mode")
            modified = True

    cmd = [sys.executable, "-m", "pytest", f"--rootdir={ROOT}"]
    if not has_asyncio:
        cmd.extend(["-m", "not asyncio"])  # skip async tests when plugin absent

    if modified:
        with NamedTemporaryFile("w", delete=False) as tmp:
            cfg.write(tmp)
            temp_ini = tmp.name
        cmd.extend(["-c", temp_ini])
    else:
        cmd.extend(["-c", str(INI_FILE)])

    for path in sorted(ignore_paths):
        cmd.append(f"--ignore={path}")

    cmd.extend(strip_xdist_flags(argv) if not has_xdist else argv)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
