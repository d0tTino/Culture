from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src import app


@pytest.mark.unit
def test_replay_detects_adjacent_event_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    event_log_path = tmp_path / "event_log.jsonl"
    event_log_path.write_text('{"type": "header", "seed": 17}\n', encoding="utf-8")

    monkeypatch.delenv("EVENT_LOG_PATH", raising=False)

    args = argparse.Namespace(
        version=False,
        scenario="Sample scenario",
        agents=3,
        steps=10,
        seed=None,
        discord=False,
        vector_store=False,
        vector_dir="./chroma_db",
        semantic_memory=False,
        semantic_db="bolt://localhost:7687",
        semantic_user="neo4j",
        semantic_password="test",
        no_warning_filters=False,
        log_suppressed_warnings=False,
        checkpoint=None,
        replay=str(snapshot),
        replay_start=None,
        replay_end=None,
        proposal=None,
        proposer_id="agent_1",
        export_dataset=None,
    )

    monkeypatch.setattr(app, "parse_args", lambda: args)
    monkeypatch.setattr(app, "setup_logging", lambda: None)

    async def _noop_load_plugins() -> None:
        return None

    monkeypatch.setattr(app, "load_plugins", _noop_load_plugins)
    monkeypatch.setattr(app, "configure_warning_filters", lambda *_, **__: None)
    monkeypatch.setattr(
        app,
        "load_scenario",
        lambda _: ("Sample scenario", None, None, [], [], {}, {}),
    )

    seed_calls: dict[str, Path | None] = {}

    def _fake_get_seed(path: Path | None = None) -> int:
        seed_calls["path"] = path
        return 42

    monkeypatch.setattr(app.event_log, "get_seed", _fake_get_seed)
    monkeypatch.setattr(app.event_log, "set_seed", lambda *_: None)

    replay_calls: dict[str, object] = {}

    def _fake_replay(snapshot_path: Path, **kwargs: object) -> None:
        replay_calls["snapshot"] = snapshot_path
        replay_calls["kwargs"] = kwargs

    monkeypatch.setattr(app.Simulation, "replay_from_snapshot", _fake_replay)

    app.main()

    assert seed_calls["path"] == event_log_path
    assert Path(replay_calls["snapshot"]) == Path(args.replay)
    assert replay_calls["kwargs"]["events_path"] == event_log_path
    assert replay_calls["kwargs"]["seed"] == 42
