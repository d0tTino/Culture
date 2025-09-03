import json
from pathlib import Path

import pytest

from src.interfaces import dashboard_backend as db


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_misbehavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"step": 1, "agent_id": "a", "reason": "bad"},
        {"step": 2, "agent_id": "b", "reason": "worse"},
    ]

    monkeypatch.setattr(db, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(db.event_log, "fetch_events", lambda event_type=None: events)


    calls: list[tuple[int, int, Path | None]] = []

    def fake_store(start: int, end: int, directory: Path | None = None) -> Path:
        calls.append((start, end, directory))
        return tmp_path / f"replay_{start}_{end}.jsonl"

    monkeypatch.setattr(db.event_log, "store_replay_slice", fake_store)

    resp = await db.api_misbehavior()
    assert json.loads(resp.body) == {
        "events": [
            {
                "step": 1,
                "agent_id": "a",
                "reason": "bad",
                "replay_path": str(tmp_path / "replay_1_1.jsonl"),
            },
            {
                "step": 2,
                "agent_id": "b",
                "reason": "worse",
                "replay_path": str(tmp_path / "replay_2_2.jsonl"),
            },
        ]
    }
    assert calls == [(1, 1, tmp_path), (2, 2, tmp_path)]

