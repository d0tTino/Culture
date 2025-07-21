import asyncio

import pytest

from src.interfaces.dashboard_backend import SimulationEvent, get_event_queue
from src.sim.simulation import Simulation


class DummyAgentState:
    def __init__(self) -> None:
        self.ip = 0.0
        self.du = 0.0
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyAgentState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: DummyAgentState) -> None:
        self.state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        return {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simulation_shutdown_cleans_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = DummyAgent("A")
    sim = Simulation([agent])

    await sim.start_event_listener()

    queue = get_event_queue()
    await queue.put(SimulationEvent(type="broadcast", data={"content": "hi"}))
    await asyncio.sleep(0.05)

    sim.close()
    await asyncio.sleep(0.05)

    pending = {task.get_coro().__name__ for task in asyncio.all_tasks() if not task.done()}

    assert "_event_listener_loop" not in pending
    assert "forward_external_events" not in pending
