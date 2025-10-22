from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.infra import event_log
from src.sim.knowledge_board import BoardEntry
from src.sim.simulation import Simulation


class DummyProducer:
    def __init__(self, store: list[bytes]):
        self.store = store

    def produce(self, _topic: str, payload: bytes) -> None:
        self.store.append(payload)

    def poll(self, _timeout: float) -> None:  # pragma: no cover - no-op
        return None


class DummyConsumer:
    def __init__(self, _conf: dict[str, object], store: list[bytes]):
        self.store = store
        self.index = 0

    def subscribe(self, _topics: list[str]) -> None:  # pragma: no cover - no-op
        pass

    def poll(self, _timeout: float):
        if self.index >= len(self.store):
            return None
        payload = self.store[self.index]
        self.index += 1

        class Message:
            def __init__(self, value: bytes) -> None:
                self._value = value

            def error(self) -> None:
                return None

            def value(self) -> bytes:
                return self._value

        return Message(payload)

    def close(self) -> None:  # pragma: no cover - no-op
        pass


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list[object]] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict[str, float]] = {}
    role: str = "dummy"
    steps_in_current_role: int = 0
    mood_level: float = 0.0

    def update_collective_metrics(self, ip: float, du: float) -> None:  # pragma: no cover - stub
        pass


class MoveAgent:
    def __init__(self, agent_id: str = "agent") -> None:
        self.agent_id = agent_id
        self.state = DummyState()
        self._added = False

    def get_id(self) -> str:  # pragma: no cover - used by scheduler
        return self.agent_id

    def update_state(self, new_state: DummyState) -> None:  # pragma: no cover - simple
        self.state = new_state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict[str, object] | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
        memory_service: object | None = None,
    ) -> dict[str, object]:
        if not self._added and knowledge_board is not None:
            knowledge_board.add_entry(
                BoardEntry(
                    content_full="hello",
                    entry_type="agent_update",
                    tags=["simulation", "movement"],
                ),
                self.agent_id,
                simulation_step,
            )
            self._added = True
        return {"map_action": {"action": "move", "dx": 1, "dy": 0}}


async def _run_simulation(tmp_path: Path) -> tuple[Simulation, Path]:
    agent = MoveAgent()
    sim = Simulation([agent])
    sim.evaluation_hooks = []

    await sim.run_step()

    snap_path = tmp_path / "snapshot_1.json"
    return sim, snap_path


@pytest.mark.asyncio
@pytest.mark.integration
async def test_replay_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_REDPANDA", "1")
    monkeypatch.setenv("SNAPSHOT_INTERVAL_STEPS", "1")

    store: list[bytes] = []
    monkeypatch.setattr(event_log, "KafkaProducer", lambda conf: DummyProducer(store))
    monkeypatch.setattr(event_log, "KafkaConsumer", lambda conf: DummyConsumer(conf, store))
    monkeypatch.setattr(event_log, "_producer", None, raising=False)

    def _save(step: int, data: dict, directory: Path = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", _save)
    monkeypatch.setattr("src.sim.simulation.upload_snapshot", lambda step: None)

    sim, snap_path = await _run_simulation(tmp_path)

    replay = Simulation.replay_from_snapshot(snap_path)

    assert replay.world_map.agent_positions == sim.world_map.agent_positions
    assert replay.knowledge_board.get_full_entries() == sim.knowledge_board.get_full_entries()
