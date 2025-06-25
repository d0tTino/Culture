"""Snapshot utilities for persisting simulation state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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


def load_snapshot(
    step: int,
    directory: str | Path = "snapshots",
    compress: bool | None = None,
) -> dict[str, Any]:
    """Load ``directory/snapshot_{step}.json`` or ``.json.zst`` and verify hash.

    Raises
    ------
    ValueError
        If the recomputed trace hash does not match the stored value.
    """

    compress = SNAPSHOT_COMPRESS if compress is None else compress

    path = Path(directory)
    json_file = path / f"snapshot_{step}.json"
    zst_file = path / f"snapshot_{step}.json.zst"
    file_path = zst_file if compress else json_file

    if not file_path.exists():
        # Fallback to the other compression option if explicit choice not found
        if compress and json_file.exists():
            file_path = json_file
            compress = False
        elif not compress and zst_file.exists():
            file_path = zst_file
            compress = True
        else:
            raise FileNotFoundError(file_path)

    if compress:
        if zstd is None:
            raise RuntimeError(
                "Compression requires the optional 'zstandard' package. Install via 'pip install zstandard'."
            )
        with file_path.open("rb") as f:
            payload = zstd.ZstdDecompressor().decompress(f.read()).decode("utf-8")
        data = json.loads(payload)
    else:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    expected = data.get("trace_hash")
    if expected is not None:
        actual = compute_trace_hash({k: v for k, v in data.items() if k != "trace_hash"})
        if actual != expected:
            raise ValueError(
                f"Trace hash mismatch for snapshot {step}: expected {expected}, computed {actual}"
            )
    return data
