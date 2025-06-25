"""Snapshot utilities for persisting simulation state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional dependency
    zstd = None

from .config import SNAPSHOT_COMPRESS


def compute_trace_hash(data: dict[str, Any]) -> str:
    """Return a stable hash for ``data`` used to verify deterministic replays."""

    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        if zstd is None:
            raise RuntimeError(
                "Compression requires the optional 'zstandard' package. Install via 'pip install zstandard'."
            )
        file_path = path / f"snapshot_{step}.json.zst"
        compressed = zstd.ZstdCompressor().compress(json.dumps(data, indent=2).encode("utf-8"))
        with file_path.open("wb") as f:
            f.write(compressed)
    else:
        file_path = path / f"snapshot_{step}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Load snapshot JSON or JSON.zst and return the data."""
    file_path = Path(path)
    if file_path.suffix == ".zst":
        if zstd is None:
            raise RuntimeError(
                "Loading compressed snapshots requires the optional 'zstandard' package."
            )
        with file_path.open("rb") as f:
            decompressed = zstd.ZstdDecompressor().decompress(f.read())
        return cast(dict[str, Any], json.loads(decompressed.decode("utf-8")))
    with file_path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))
