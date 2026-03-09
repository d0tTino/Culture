from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.infra.snapshot import load_snapshot as _load_snapshot
from src.infra.snapshot import save_snapshot as _save_snapshot
from src.infra.snapshot import upload_snapshot as _upload_snapshot
from src.sim.persistence.snapshot_migrations import CURRENT_SNAPSHOT_SCHEMA_VERSION

from .trace_hash_service import TraceHashService


class SnapshotPersistenceService:
    """Snapshot/replay-oriented persistence service over infra storage primitives."""

    @staticmethod
    def _validate_schema_version(data: dict[str, Any]) -> None:
        version = data.get("snapshot_schema_version")
        if not isinstance(version, int):
            raise ValueError("Snapshot schema version is missing or invalid")
        if version < 1 or version > CURRENT_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported snapshot schema version {version}; "
                f"supported range is [1, {CURRENT_SNAPSHOT_SCHEMA_VERSION}]"
            )

    @staticmethod
    def normalize_for_hash(data: dict[str, Any]) -> dict[str, Any]:
        normalized = {k: v for k, v in data.items() if k != "trace_hash"}
        knowledge_board = normalized.get("knowledge_board")
        if isinstance(knowledge_board, dict):
            normalized["knowledge_board"] = {
                k: v for k, v in knowledge_board.items() if k != "vector"
            }
        world_map = normalized.get("world_map")
        if isinstance(world_map, dict):
            normalized["world_map"] = {k: v for k, v in world_map.items() if k != "vector"}
        return normalized

    @classmethod
    def compute_hash(cls, data: dict[str, Any]) -> str:
        return TraceHashService.compute(cls.normalize_for_hash(data))

    @classmethod
    def validate(cls, step: int | str | Path, data: dict[str, Any]) -> dict[str, Any]:
        cls._validate_schema_version(data)
        expected = data.get("trace_hash")
        if expected is None:
            return data
        actual = cls.compute_hash(data)
        if actual != expected:
            raise ValueError(
                f"Trace hash mismatch for snapshot {step}: expected {expected}, computed {actual}"
            )
        return data

    @staticmethod
    def save(step: int, data: dict[str, Any], directory: str | Path = "snapshots") -> None:
        SnapshotPersistenceService._validate_schema_version(data)
        _save_snapshot(step, data, directory=directory)

    @staticmethod
    def latest_snapshot_path(directory: str | Path = "snapshots") -> Path | None:
        path = Path(directory)
        if not path.exists():
            return None

        latest: tuple[int, Path] | None = None
        for candidate in path.iterdir():
            if not candidate.is_file():
                continue
            match = re.match(r"^snapshot_(\d+)\.json(?:\.zst)?$", candidate.name)
            if match is None:
                continue
            step = int(match.group(1))
            if latest is None or step > latest[0]:
                latest = (step, candidate)
        return latest[1] if latest is not None else None

    @classmethod
    def load_latest(cls, directory: str | Path = "snapshots") -> tuple[Path, dict[str, Any]] | None:
        latest_path = cls.latest_snapshot_path(directory=directory)
        if latest_path is None:
            return None
        return latest_path, cls.load(latest_path, directory=directory)

    @staticmethod
    def upload(step: int, directory: str | Path = "snapshots") -> None:
        _upload_snapshot(step, directory=directory)

    @staticmethod
    def load(step: int | str | Path, directory: str | Path = "snapshots") -> dict[str, Any]:
        data = _load_snapshot(step, directory=directory)
        return SnapshotPersistenceService.validate(step, data)
