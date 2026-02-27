import json
from pathlib import Path

import pytest

from src.infra.snapshot import save_snapshot
from src.interfaces import dashboard_backend as db
from src.sim.persistence.snapshot_service import SnapshotPersistenceService


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_memory_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # create several snapshot files
    steps = [1, 2, 3]
    for step in steps:
        data = {"step": step, "snapshot_schema_version": 3}
        data["trace_hash"] = SnapshotPersistenceService.compute_hash(data)
        save_snapshot(step, data, directory=tmp_path)
    monkeypatch.setattr(db, "SNAPSHOT_DIR", tmp_path)

    resp = await db.api_memory_snapshots(limit=2)
    assert json.loads(resp.body) == {"steps": steps[-2:]}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_memory_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"step": 5, "snapshot_schema_version": 3}
    data["trace_hash"] = SnapshotPersistenceService.compute_hash(data)
    save_snapshot(5, data, directory=tmp_path)
    monkeypatch.setattr(db, "SNAPSHOT_DIR", tmp_path)

    resp = await db.api_memory_snapshot(5)
    assert json.loads(resp.body) == data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_memory_snapshot_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(db, "SNAPSHOT_DIR", tmp_path)

    resp = await db.api_memory_snapshot(42)
    assert resp.status_code == 404
    assert json.loads(resp.body) == {"error": "not_found"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_flagged_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "flagged_message", "step": 1, "message": "bad"},
        {"type": "flagged_message", "step": 2, "message": "worse"},
    ]

    monkeypatch.setattr(db, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(db.event_log, "fetch_events", lambda *a, **k: events)

    calls: list[tuple[int, int, Path | None]] = []

    def fake_store(start: int, end: int, directory: Path | None = None) -> Path:
        calls.append((start, end, directory))
        return tmp_path / f"replay_{start}_{end}.jsonl"

    monkeypatch.setattr(db.event_log, "store_replay_slice", fake_store)

    resp = await db.api_flagged_messages()
    assert json.loads(resp.body) == {
        "messages": [
            {"step": 1, "message": "bad", "snapshot": "snapshot_1.json"},
            {"step": 2, "message": "worse", "snapshot": "snapshot_2.json"},
        ]
    }
    assert calls == [(1, 1, tmp_path), (2, 2, tmp_path)]
