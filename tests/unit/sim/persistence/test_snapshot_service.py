from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infra.snapshot import save_snapshot
from src.sim.persistence.snapshot_migrations import CURRENT_SNAPSHOT_SCHEMA_VERSION
from src.sim.persistence.snapshot_service import SnapshotPersistenceService

pytestmark = pytest.mark.unit


def _snapshot_payload(version: int) -> dict[str, object]:
    payload: dict[str, object] = {
        "step": 7,
        "snapshot_schema_version": version,
        "knowledge_board": {"entries": [{"id": "k1"}], "vector": {"clock": 1}},
        "world_map": {"width": 10, "height": 10, "agents": {}, "vector": {"clock": 2}},
    }
    payload["trace_hash"] = SnapshotPersistenceService.compute_hash(payload)
    return payload


def test_load_snapshot_validates_current_schema(tmp_path: Path) -> None:
    payload = _snapshot_payload(CURRENT_SNAPSHOT_SCHEMA_VERSION)
    save_snapshot(7, payload, directory=tmp_path)

    loaded = SnapshotPersistenceService.load(7, directory=tmp_path)

    assert loaded == payload


def test_load_snapshot_accepts_previous_schema(tmp_path: Path) -> None:
    payload = _snapshot_payload(CURRENT_SNAPSHOT_SCHEMA_VERSION - 1)
    save_snapshot(7, payload, directory=tmp_path)

    loaded = SnapshotPersistenceService.load(7, directory=tmp_path)

    assert loaded["snapshot_schema_version"] == CURRENT_SNAPSHOT_SCHEMA_VERSION - 1


def test_load_snapshot_rejects_tampered_hash(tmp_path: Path) -> None:
    payload = _snapshot_payload(CURRENT_SNAPSHOT_SCHEMA_VERSION)
    save_snapshot(7, payload, directory=tmp_path)

    path = tmp_path / "snapshot_7.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["step"] = 8
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Trace hash mismatch"):
        SnapshotPersistenceService.load(7, directory=tmp_path)
