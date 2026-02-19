from __future__ import annotations

from pathlib import Path
from typing import Any

from src.infra.snapshot import load_snapshot as _load_snapshot
from src.infra.snapshot import save_snapshot as _save_snapshot
from src.infra.snapshot import upload_snapshot as _upload_snapshot

from .trace_hash_service import TraceHashService


class SnapshotPersistenceService:
    """Snapshot/replay-oriented persistence service over infra storage primitives."""

    @staticmethod
    def save(step: int, data: dict[str, Any], directory: str | Path = "snapshots") -> None:
        _save_snapshot(step, data, directory=directory)

    @staticmethod
    def upload(step: int, directory: str | Path = "snapshots") -> None:
        _upload_snapshot(step, directory=directory)

    @staticmethod
    def load(step: int | str | Path, directory: str | Path = "snapshots") -> dict[str, Any]:
        data = _load_snapshot(step, directory=directory)
        expected = data.get("trace_hash")
        if expected is None:
            return data
        data_no_vector = {k: v for k, v in data.items() if k != "trace_hash"}
        if "knowledge_board" in data_no_vector:
            data_no_vector["knowledge_board"] = {
                k: v for k, v in data.get("knowledge_board", {}).items() if k != "vector"
            }
        if "world_map" in data_no_vector:
            data_no_vector["world_map"] = {
                k: v for k, v in data.get("world_map", {}).items() if k != "vector"
            }
        actual = TraceHashService.compute(data_no_vector)
        if actual != expected:
            raise ValueError(
                f"Trace hash mismatch for snapshot {step}: expected {expected}, computed {actual}"
            )
        return data
