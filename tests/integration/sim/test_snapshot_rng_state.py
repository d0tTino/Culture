import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.infra import event_log
from src.sim.simulation import Simulation


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    mood_level: float = 0.0


class RandomAgent:
    def __init__(self, agent_id: str = "agent") -> None:
        self.agent_id = agent_id
        self.state = DummyState()

    def get_id(self) -> str:  # pragma: no cover - used by scheduler
        return self.agent_id

    def update_state(self, new_state: DummyState) -> None:  # pragma: no cover - simple
        self.state = new_state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
        memory_service: object | None = None,
    ) -> dict:
        self.state.ip = random.random()
        return {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_snapshot_rng_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr("src.agents.core.base_agent.Agent", RandomAgent)
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda step: None)

    def _save(step: int, data: dict, directory: Path = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", _save)

    event_log._last_hash = None
    event_log._seed = None
    seed = 1234
    random.seed(seed)
    event_log.set_seed(seed)

    sim = Simulation([RandomAgent()], seed=seed)
    await sim.run_step()  # step 1
    await sim.run_step()  # step 2
    snap_path = tmp_path / "snapshot_2.json"

    # Continue original run to step 3
    await sim.run_step()
    original_ip = sim.agents[0].state.ip

    # Replay from snapshot_2 using the stored seed and event log
    event_log._last_hash = None
    event_log._seed = None
    stored = event_log.get_seed(str(tmp_path / "events.jsonl")) or seed
    replay = Simulation.replay_from_snapshot(
        snap_path, start_step=3, end_step=3, seed=stored
    )
    replay_ip = replay.agents[0].state.ip

    assert replay_ip == original_ip
