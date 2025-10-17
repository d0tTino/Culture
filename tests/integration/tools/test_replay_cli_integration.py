from __future__ import annotations

import pytest

from src.sim.simulation import Simulation
from tests.utils.replay_cli_fixture import (
    ReplayFixtureAgent,
    ReplayFixtureState,
    create_replay_cli_artifacts,
)
from tools import replay_cli


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replay_cli_replays_fixture(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = await create_replay_cli_artifacts(tmp_path, monkeypatch)

    captured: list[Simulation] = []
    original = Simulation.replay_from_snapshot.__func__

    def fake_from_snapshot(
        cls: type[Simulation], snapshot: dict, *, seed: int | None = None
    ) -> Simulation:
        agents: list[ReplayFixtureAgent] = []
        agents_data = snapshot.get("agents", [])
        if not agents_data:
            agents_data = [{"agent_id": "agent-1"}]
        for idx, data in enumerate(agents_data):
            agent_id = data.get("agent_id", f"agent-{idx}")
            agent = ReplayFixtureAgent(agent_id)
            state = ReplayFixtureState()
            state.ip = float(data.get("ip", 0.0))
            state.du = float(data.get("du", 0.0))
            agent.state = state
            agents.append(agent)

        sim = Simulation(agents=agents)
        sim.current_step = int(snapshot.get("step", 0))
        sim.collective_ip = float(snapshot.get("collective_ip", 0.0))
        sim.collective_du = float(snapshot.get("collective_du", 0.0))

        kb = snapshot.get("knowledge_board", {})
        entries = kb.get("entries", [])
        sim.knowledge_board.entries.clear()
        sim.knowledge_board.entries.extend(entries)

        world_map = snapshot.get("world_map", {})
        if isinstance(world_map, dict):
            sim.world_map.width = int(world_map.get("width", sim.world_map.width))
            sim.world_map.height = int(world_map.get("height", sim.world_map.height))
            agents_positions = world_map.get("agents", {})
            sim.world_map.agent_positions = {
                key: tuple(value) if isinstance(value, (list, tuple)) else value
                for key, value in agents_positions.items()
            }
        return sim

    monkeypatch.setattr(
        replay_cli.Simulation, "from_snapshot", classmethod(fake_from_snapshot)
    )

    def capture_replay(
        cls: type[Simulation],
        snapshot,
        *,
        start_step=None,
        end_step=None,
        seed=None,
        events_path=None,
    ) -> Simulation:
        sim = original(
            cls,
            snapshot,
            start_step=start_step,
            end_step=end_step,
            seed=seed,
            events_path=events_path,
        )
        captured.append(sim)
        return sim

    monkeypatch.setattr(
        replay_cli.Simulation, "replay_from_snapshot", classmethod(capture_replay)
    )

    exit_code = replay_cli.main(
        [
            str(artifacts.snapshot_path),
            "--from",
            str(artifacts.start_step),
            "--to",
            str(artifacts.end_step),
            "--events",
            str(artifacts.replay_slice_path),
        ]
    )

    assert exit_code == 0
    assert captured, "expected the replay CLI to invoke Simulation.replay_from_snapshot"

    replayed = captured[-1]
    try:
        assert replayed.current_step == artifacts.expected_step
        assert (
            replayed.knowledge_board.to_dict().get("entries", [])
            == artifacts.expected_knowledge_entries
        )
    finally:
        replayed.close()
