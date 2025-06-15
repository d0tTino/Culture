"""Snapshot utilities for persisting simulation state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_snapshot(step: int, data: dict[str, Any], directory: str | Path = "snapshots") -> None:
    """Serialize ``data`` to ``directory/snapshot_{step}.json``.

    Parameters
    ----------
    step:
        Current simulation step.
    data:
        Dictionary containing snapshot information.
    directory:
        Folder where snapshots will be stored.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"snapshot_{step}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
