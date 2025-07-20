import asyncio

import pytest

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
async def test_listener_tasks_reused_across_steps() -> None:
    agent = DummyAgent("A")
    sim = Simulation([agent])

    for _ in range(3):
        await sim.run_step()

    pending = [t.get_coro().__name__ for t in asyncio.all_tasks() if not t.done()]
    assert pending.count("_event_listener_loop") <= 1
    assert pending.count("forward_external_events") <= 1

    await sim.stop_event_listener()
    sim.close()
