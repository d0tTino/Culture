import pytest

from tools import replay_cli


@pytest.mark.unit
def test_replay_cli_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def mock_replay(snapshot: str, start_step=None, end_step=None, seed=None):
        called.update(snapshot=snapshot, start=start_step, end=end_step, seed=seed)

    monkeypatch.setattr(replay_cli.Simulation, "replay_from_snapshot", mock_replay)

    assert (
        replay_cli.main(["snap.json", "--from", "2", "--to", "4", "--seed", "99"]) == 0
    )
    assert called == {
        "snapshot": "snap.json",
        "start": 2,
        "end": 4,
        "seed": 99,
    }


@pytest.mark.unit
def test_replay_cli_tick_range_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def mock_replay(snapshot: str, start_step=None, end_step=None, seed=None):
        called.update(start=start_step, end=end_step)

    monkeypatch.setattr(replay_cli.Simulation, "replay_from_snapshot", mock_replay)

    replay_cli.main(["snap.json", "--from-tick", "5", "--to-tick", "7"])
    assert called["start"] == 5
    assert called["end"] == 7
