#!/usr/bin/env python3
"""Run pytest with optional plugin-based adjustments."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent.parent
INI_FILE = ROOT / "pytest.ini"


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main(argv: list[str]) -> int:
    has_xdist = have_module("xdist")
    has_asyncio = have_module("pytest_asyncio")
    # Install dependencies if any are missing unless skipped via env var
    required = [
        "fastapi",
        "sqlalchemy",
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

    cmd = [sys.executable, "-m", "pytest"]

    if modified:
        with NamedTemporaryFile("w", delete=False) as tmp:
            cfg.write(tmp)
            temp_ini = tmp.name
        cmd.extend(["-c", temp_ini])
    else:
        cmd.extend(["-c", str(INI_FILE)])

    cmd.extend(argv)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
