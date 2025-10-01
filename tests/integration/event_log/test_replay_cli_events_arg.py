"""Integration tests for replay CLI event log selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app import create_simulation
from src.infra import event_log
from src.infra.snapshot import compute_trace_hash, save_snapshot
from src.sim.simulation import Simulation
from tests.utils.mock_llm import MockLLM
from tools import replay_cli


def _snapshot_from_sim(sim: Simulation) -> dict:
    snapshot = {
        "step": sim.current_step,
        "collective_ip": sim.collective_ip,
        "collective_du": sim.collective_du,
        "knowledge_board": sim.knowledge_board.to_dict(),
        "world_map": sim.world_map.to_dict(),
        "agents": [
            {
                "agent_id": ag.agent_id,
                "ip": ag.state.ip,
                "du": ag.state.du,
            }
            for ag in sim.agents
        ],
    }
    snapshot_no_vector = {
        **{k: v for k, v in snapshot.items() if k != "trace_hash"},
        "knowledge_board": {k: v for k, v in snapshot["knowledge_board"].items() if k != "vector"},
        "world_map": {k: v for k, v in snapshot["world_map"].items() if k != "vector"},
    }
    snapshot["trace_hash"] = compute_trace_hash(snapshot_no_vector)
    return snapshot


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replay_cli_events_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The replay CLI should accept ``--events`` and discover logs by default."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENABLE_REDPANDA", "0")
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    monkeypatch.delenv("EVENT_LOG_PATH", raising=False)
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda *a, **k: None)

    def save(step: int, data: dict, directory: Path = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=2, scenario="test")
        sim.evaluation_hooks = []
        base_snapshot = _snapshot_from_sim(sim)
        save_snapshot(0, base_snapshot, directory=tmp_path)
        await sim.run_step()
    sim.close()

    snapshot_path = tmp_path / "snapshot_0.json"
    events_path = tmp_path / "event_log.jsonl"
    assert snapshot_path.exists()
    assert events_path.exists()

    last_logged_step = 0
    agent_action_steps: list[int] = []
    with events_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "header":
                continue
            step = int(event.get("step", event.get("tick", 0)))
            last_logged_step = max(last_logged_step, step)
            if event.get("type") == "agent_action":
                agent_action_steps.append(step)

    assert last_logged_step >= 1
    assert agent_action_steps

    applied_runs: list[list[int]] = []
    original_apply = Simulation.apply_event

    def capturing_apply(self, event):
        if isinstance(event, dict) and event.get("type") == "agent_action":
            if not applied_runs:
                applied_runs.append([])
            applied_runs[-1].append(int(event.get("step", event.get("tick", 0))))
        return original_apply(self, event)

    monkeypatch.setattr(Simulation, "apply_event", capturing_apply)

    original_stream = event_log.stream_events
    stream_paths: list[Path | None] = []

    def capture_stream(
        after_step: int = 0,
        timeout: float = 1.0,
        *,
        end_step: int | None = None,
        path: str | Path | None = None,
    ):
        stream_paths.append(Path(path) if path is not None else None)
        yield from original_stream(
            after_step=after_step, timeout=timeout, end_step=end_step, path=path
        )

    monkeypatch.setattr(event_log, "stream_events", capture_stream)

    original_replay = Simulation.replay_from_snapshot.__func__

    def closing_replay(
        cls: type[Simulation],
        snapshot: Path,
        *,
        start_step=None,
        end_step=None,
        seed=None,
        events_path=None,
    ) -> Simulation:
        sim = original_replay(
            cls,
            snapshot,
            start_step=start_step,
            end_step=end_step,
            seed=seed,
            events_path=events_path,
        )
        sim.close()
        return sim

    monkeypatch.setattr(
        replay_cli.Simulation, "replay_from_snapshot", classmethod(closing_replay)
    )

    from_tick = "1" if last_logged_step >= 1 else "0"
    to_tick = str(last_logged_step)

    applied_runs.append([])
    prev_stream_len = len(stream_paths)
    assert (
        replay_cli.main(
            [
                str(snapshot_path),
                "--from",
                from_tick,
                "--to",
                to_tick,
                "--events",
                str(events_path),
            ]
        )
        == 0
    )
    assert len(stream_paths) == prev_stream_len + 1
    assert stream_paths[-1] == events_path
    assert applied_runs[-1] == agent_action_steps

    monkeypatch.delenv("EVENT_LOG_PATH", raising=False)
    applied_runs.append([])
    prev_stream_len = len(stream_paths)
    assert (
        replay_cli.main([str(snapshot_path), "--from", from_tick, "--to", to_tick])
        == 0
    )
    assert len(stream_paths) == prev_stream_len + 1
    assert stream_paths[-1] == events_path
    assert applied_runs[-1] == agent_action_steps
