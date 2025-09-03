import json
from pathlib import Path

import pytest

from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_misbehavior_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "misbehavior", "step": 1, "detail": "bad"},
        {"type": "misbehavior", "step": 2, "detail": "worse"},
        {"type": "misbehavior", "step": 3, "detail": "awful"},
    ]

    monkeypatch.setattr(db.event_log, "fetch_events", lambda *a, **k: events)
    monkeypatch.setattr(db, "SNAPSHOT_DIR", tmp_path)

    calls: list[tuple[int, int, Path | None]] = []

    def fake_store(start: int, end: int, directory: Path | None = None) -> Path:
        calls.append((start, end, directory))
        return tmp_path / f"replay_{start}_{end}.jsonl"

    monkeypatch.setattr(db.event_log, "store_replay_slice", fake_store)

    resp = await db.api_misbehavior(limit=2)
    assert json.loads(resp.body) == {
        "events": [
            {"step": 2, "detail": "worse", "replay": "replay_2_2.jsonl"},
            {"step": 3, "detail": "awful", "replay": "replay_3_3.jsonl"},
        ]
    }
    assert calls == [(2, 2, tmp_path), (3, 3, tmp_path)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_misbehavior_non_positive_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [{"type": "misbehavior", "step": 1, "detail": "bad"}]

    monkeypatch.setattr(db.event_log, "fetch_events", lambda *a, **k: events)
    monkeypatch.setattr(db, "SNAPSHOT_DIR", tmp_path)

    calls: list[tuple[int, int, Path | None]] = []

    def fake_store(start: int, end: int, directory: Path | None = None) -> Path:
        calls.append((start, end, directory))
        return tmp_path / f"replay_{start}_{end}.jsonl"

    monkeypatch.setattr(db.event_log, "store_replay_slice", fake_store)

    resp = await db.api_misbehavior(limit=0)
    assert json.loads(resp.body) == {"events": []}
    assert calls == []
