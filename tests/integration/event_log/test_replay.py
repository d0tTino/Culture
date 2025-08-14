from pathlib import Path

import pytest

from src.app import create_simulation
from src.infra.snapshot import compute_trace_hash, save_snapshot
from src.sim.simulation import Simulation
from tests.utils.mock_llm import MockLLM


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
                "mood": ag.state.mood_level,
            }
            for ag in sim.agents
        ],
    }
    snapshot_no_vector = {
        **{k: v for k, v in snapshot.items() if k != "trace_hash"},
        "knowledge_board": {
            k: v for k, v in snapshot["knowledge_board"].items() if k != "vector"
        },
        "world_map": {k: v for k, v in snapshot["world_map"].items() if k != "vector"},
    }
    snapshot["trace_hash"] = compute_trace_hash(snapshot_no_vector)
    return snapshot


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replay_matches_trace_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    log_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_path))
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda *a, **k: None)

    def save(step: int, data: dict, directory: Path = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", save)

    with MockLLM():
        sim = create_simulation(num_agents=1, steps=1, scenario="test")
        snap0 = _snapshot_from_sim(sim)
        save_snapshot(0, snap0, directory=tmp_path)
        await sim.run_step()
        snap1 = _snapshot_from_sim(sim)

    expected_hash = snap1["trace_hash"]
    replay = Simulation.replay_from_snapshot(
        tmp_path / "snapshot_0.json", start_step=1, end_step=1
    )
    replay_snap = _snapshot_from_sim(replay)
    assert replay_snap["trace_hash"] == expected_hash
