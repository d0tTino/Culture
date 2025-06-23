import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest

from src.agents.core.base_agent import Agent
from src.interfaces.dashboard_backend import SimulationEvent, event_queue


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


sys.modules.setdefault("neo4j", DummyNeo4j())

from src.sim.simulation import Simulation


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
    short_term_memory: ClassVar[list[Any]] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict[str, float]] = {}
    role: str = "dummy"
    steps_in_current_role: int = 0
    mood_level: float = 0.0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        pass


class MoveAgent:
    def __init__(self, agent_id: str = "agent") -> None:
        self.agent_id = agent_id
        self.state = DummyState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict[str, Any] | None = None,
        vector_store_manager: Any | None = None,
        knowledge_board: Any | None = None,
    ) -> dict[str, Any]:
        return {"map_action": {"action": "move", "dx": 1, "dy": 0}}


async def _clear_event_queue() -> None:
    while not event_queue.empty():
        _ = await event_queue.get()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_move_updates_map_and_events() -> None:
    await _clear_event_queue()
    agent = MoveAgent()
    sim = Simulation(agents=cast(list[Agent], [agent]))

    await sim.run_step()

    assert sim.world_map.agent_positions[agent.agent_id] == (1, 0)

    evt = await asyncio.wait_for(event_queue.get(), 0.1)
    assert isinstance(evt, SimulationEvent)
    assert evt.event_type == "map_action"
    assert evt.data is not None
    assert evt.data["agent_id"] == agent.agent_id


@pytest.mark.asyncio
@pytest.mark.integration
async def test_snapshot_contains_world_map(tmp_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MoveAgent()
    sim = Simulation(agents=cast(list[Agent], [agent]))

    def _save(step: int, data: dict[str, Any], directory: str = tmp_path) -> None:
        from src.infra.snapshot import save_snapshot as real_save

        real_save(step, data, directory)

    monkeypatch.setattr("src.sim.simulation.save_snapshot", _save)
    monkeypatch.setattr("src.sim.simulation.log_event", lambda event: None)

    for _ in range(100):
        await sim.run_step()

    snap_file = f"{tmp_path}/snapshot_100.json"
    with open(snap_file) as f:
        snapshot = json.load(f)

    assert "world_map" in snapshot
    assert snapshot["world_map"]["agents"][agent.agent_id] == [9, 0]
