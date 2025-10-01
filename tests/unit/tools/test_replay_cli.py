from pathlib import Path

import pytest

from tools import replay_cli


@pytest.mark.unit
def test_replay_cli_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def mock_replay(snapshot: str, start_step=None, end_step=None, seed=None, event_log_path=None):
        called.update(
            snapshot=snapshot, start=start_step, end=end_step, seed=seed, events=event_log_path
        )

    monkeypatch.setattr(replay_cli.Simulation, "replay_from_snapshot", mock_replay)

    assert (
        replay_cli.main(["snap.json", "--from", "2", "--to", "4", "--seed", "99"]) == 0
    )
    assert called == {
        "snapshot": "snap.json",
        "start": 2,
        "end": 4,
        "seed": 99,
        "events": None,
    }


@pytest.mark.unit
def test_replay_cli_tick_range_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def mock_replay(snapshot: str, start_step=None, end_step=None, seed=None, event_log_path=None):
        called.update(start=start_step, end=end_step, events=event_log_path)

    monkeypatch.setattr(replay_cli.Simulation, "replay_from_snapshot", mock_replay)

    replay_cli.main(["snap.json", "--from-tick", "5", "--to-tick", "7"])
    assert called["start"] == 5
    assert called["end"] == 7
    assert called["events"] is None


@pytest.mark.unit
def test_replay_cli_discovers_event_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = tmp_path / "event_log.jsonl"
    events.write_text("{\"type\": \"header\", \"seed\": 123}\n", encoding="utf-8")
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    snapshot = snapshot_dir / "snapshot_0.json"
    snapshot.write_text("{}", encoding="utf-8")

    captured_seed: dict[str, object] = {}

    def fake_get_seed(path=None):
        captured_seed["path"] = path
        return 321

    called: dict[str, object] = {}

    def mock_replay(snapshot: str, start_step=None, end_step=None, seed=None, event_log_path=None):
        called.update(seed=seed, events=event_log_path)

    monkeypatch.setattr(replay_cli.event_log, "get_seed", fake_get_seed)
    monkeypatch.setattr(replay_cli.Simulation, "replay_from_snapshot", mock_replay)

    replay_cli.main([str(snapshot), "--from", "1", "--to", "2"])

    assert captured_seed["path"] == events
    assert called == {"seed": 321, "events": events}


@pytest.mark.unit
def test_replay_cli_events_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events = tmp_path / "custom.jsonl"
    events.write_text("{\"type\": \"header\", \"seed\": 555}\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot_0.json"
    snapshot.write_text("{}", encoding="utf-8")

    captured_seed: dict[str, object] = {}

    def fake_get_seed(path=None):
        captured_seed["path"] = path
        return 555

    called: dict[str, object] = {}

    def mock_replay(snapshot: str, start_step=None, end_step=None, seed=None, event_log_path=None):
        called.update(snapshot=snapshot, events=event_log_path, seed=seed)

    monkeypatch.setattr(replay_cli.event_log, "get_seed", fake_get_seed)
    monkeypatch.setattr(replay_cli.Simulation, "replay_from_snapshot", mock_replay)

    replay_cli.main([
        str(snapshot),
        "--from",
        "3",
        "--to",
        "4",
        "--events",
        str(events),
    ])

    assert captured_seed["path"] == events
    assert called == {"snapshot": str(snapshot), "events": events, "seed": 555}
