"""Snapshot utilities for persisting simulation state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import zstandard as zstd

from .config import SNAPSHOT_COMPRESS


def save_snapshot(
    step: int,
    data: dict[str, Any],
    directory: str | Path = "snapshots",
    compress: bool | None = None,
) -> None:
    """Serialize ``data`` to ``directory/snapshot_{step}.json`` or ``.json.zst``.

    Parameters
    ----------
    step:
        Current simulation step.
    data:
        Dictionary containing snapshot information.
    directory:
        Folder where snapshots will be stored.
    """
    compress = SNAPSHOT_COMPRESS if compress is None else compress

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    if compress:
        file_path = path / f"snapshot_{step}.json.zst"
        compressed = zstd.ZstdCompressor().compress(json.dumps(data, indent=2).encode("utf-8"))
        with file_path.open("wb") as f:
            f.write(compressed)
    else:
        file_path = path / f"snapshot_{step}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
