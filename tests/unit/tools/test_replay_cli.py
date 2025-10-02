from pathlib import Path

import pytest

from tools import replay_cli


@pytest.mark.unit
def test_replay_cli_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def mock_replay(
        snapshot: Path,
        *,
        start_step=None,
        end_step=None,
        seed=None,
        events_path=None,
    ):
        called.update(
            snapshot=snapshot,
            start=start_step,
            end=end_step,
            seed=seed,
            events=events_path,

        )

    monkeypatch.setattr(replay_cli.Simulation, "replay_from_snapshot", mock_replay)

    assert (
        replay_cli.main(["snap.json", "--from", "2", "--to", "4", "--seed", "99"]) == 0
    )
    assert called == {
        "snapshot": Path("snap.json"),
        "start": 2,
        "end": 4,
        "seed": 99,
        "events": None,
    }


@pytest.mark.unit
def test_replay_cli_tick_range_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def mock_replay(
        snapshot: Path,
        *,
        start_step=None,
        end_step=None,
        seed=None,
        events_path=None,
    ):
        called.update(start=start_step, end=end_step, events=events_path)


    monkeypatch.setattr(replay_cli.Simulation, "replay_from_snapshot", mock_replay)

    replay_cli.main(["snap.json", "--from-tick", "5", "--to-tick", "7"])
    assert called["start"] == 5
    assert called["end"] == 7
    assert called["events"] is None


@pytest.mark.unit
def test_resolve_event_log_handles_compressed_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "sim.json.zst"
    snapshot.touch()
    expected = tmp_path / "sim.jsonl"
    expected.touch()

    resolved = replay_cli._resolve_event_log(snapshot, explicit=None)

    assert resolved == expected
