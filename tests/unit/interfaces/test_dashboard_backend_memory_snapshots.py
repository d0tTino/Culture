import json
from pathlib import Path

import pytest

from src.infra.snapshot import compute_trace_hash, save_snapshot
from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_memory_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # create several snapshot files
    steps = [1, 2, 3]
    for step in steps:
        data = {"step": step}
        data["trace_hash"] = compute_trace_hash(data)
        save_snapshot(step, data, directory=tmp_path)
    monkeypatch.setattr(db, "SNAPSHOT_DIR", tmp_path)

    resp = await db.api_memory_snapshots(limit=2)
    assert json.loads(resp.body) == {"steps": steps[-2:]}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_memory_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"step": 5}
    data["trace_hash"] = compute_trace_hash(data)
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
