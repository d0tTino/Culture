import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.sim.simulation import Simulation


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict] = {}
    role: str = "dummy"
    steps_in_current_role: int = 0
    mood_level: float = 0.0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        pass


class DummyAgent:
    def __init__(self, agent_id: str = "dummy") -> None:
        self.agent_id = agent_id
        self.state = DummyState()
        self._added = False

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        if not self._added and knowledge_board is not None:
            knowledge_board.add_entry("test entry", self.agent_id, simulation_step)
            self._added = True
        return {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_snapshot_contains_knowledge_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = DummyAgent()
    sim = Simulation(agents=[agent])

    def _save(step: int, data: dict, directory: Path = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", _save)
    monkeypatch.setattr("src.sim.simulation.log_event", lambda event: None)

    for _ in range(100):
        await sim.run_step()

    snap_file = tmp_path / "snapshot_100.json"
    with snap_file.open() as f:
        snapshot = json.load(f)

    assert "knowledge_board" in snapshot
    assert snapshot["knowledge_board"]["entries"]
    assert "trace_hash" in snapshot

    from src.infra.snapshot import compute_trace_hash

    expected_hash = compute_trace_hash({k: v for k, v in snapshot.items() if k != "trace_hash"})
    assert snapshot["trace_hash"] == expected_hash

    agent2 = DummyAgent()
    sim2 = Simulation(agents=[agent2])

    def _save2(step: int, data: dict, directory: Path = tmp_path / "run2") -> None:
        from src.infra.snapshot import save_snapshot as real_save

        (tmp_path / "run2").mkdir(exist_ok=True)
        real_save(step, data, tmp_path / "run2")

    monkeypatch.setattr("src.sim.simulation.save_snapshot", _save2)

    for _ in range(100):
        await sim2.run_step()

    snap_file2 = tmp_path / "run2" / "snapshot_100.json"
    with snap_file2.open() as f2:
        snapshot2 = json.load(f2)

    assert snapshot2["trace_hash"] == snapshot["trace_hash"]
    assert snapshot2["knowledge_board"] == snapshot["knowledge_board"]
