import asyncio
import sys
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

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
    relationships: ClassVar[dict[str, Any]] = {}
    role: str = "dummy"
    steps_in_current_role: int = 0

    def update_collective_metrics(self, ip: float, du: float) -> None:
        pass


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
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
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"step": simulation_step}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_agent_turns() -> None:
    agents = [DummyAgent(str(i)) for i in range(5)]
    sim = Simulation(agents=agents)  # type: ignore[arg-type]

    start = time.perf_counter()
    results = await sim.run_turns_concurrent(agents)
    elapsed = time.perf_counter() - start

    assert len(results) == 5
    assert elapsed < 0.5
    assert sim.current_step == 5


async def _clear_event_queue() -> None:
    while not event_queue.empty():
        _ = await event_queue.get()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_events_enqueued_during_run_step() -> None:
    await _clear_event_queue()
    agent = DummyAgent("agent1")
    sim = Simulation(agents=[agent])  # type: ignore[list-item]

    await sim.run_step()

    evt = await asyncio.wait_for(event_queue.get(), 0.1)
    assert isinstance(evt, SimulationEvent)
    assert evt.event_type == "agent_action"
    assert evt.data is not None
    assert evt.data["agent_id"] == "agent1"
    assert evt.data["step"] == 1
    await _clear_event_queue()
