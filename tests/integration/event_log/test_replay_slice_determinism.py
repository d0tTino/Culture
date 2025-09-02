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
async def test_replay_slice_determinism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_REDPANDA", "0")
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda *a, **k: None)

    def save(step: int, data: dict, directory: Path = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="test")
        sim.evaluation_hooks = []
        snap0 = _snapshot_from_sim(sim)
        save_snapshot(0, snap0, directory=tmp_path)
        for _ in range(3):
            await sim.run_step()

    slice_path = event_log.store_replay_slice(1, 3, directory=tmp_path)

    expected_actions = []
    with slice_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            ev = json.loads(line)
            if ev.get("type") == "agent_action":
                expected_actions.append(
                    {
                        "step": ev.get("step"),
                        "agent_id": ev.get("agent_id"),
                        "ip": ev.get("ip"),
                        "du": ev.get("du"),
                    }
                )

    applied_actions = []
    original_apply = replay_cli.Simulation.apply_event

    def capturing_apply(self, event):
        if event.get("type") == "agent_action":
            applied_actions.append(
                {
                    "step": event.get("step"),
                    "agent_id": event.get("agent_id"),
                    "ip": event.get("ip"),
                    "du": event.get("du"),
                }
            )
        return original_apply(self, event)

    monkeypatch.setattr(replay_cli.Simulation, "apply_event", capturing_apply)

    monkeypatch.setenv("EVENT_LOG_PATH", str(slice_path))
    snapshot_path = tmp_path / "snapshot_0.json"
    replay_cli.main([str(snapshot_path), "--from", "1", "--to", "3"])

    assert applied_actions == expected_actions
