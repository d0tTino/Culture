import asyncio
import time
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
        environment_perception: dict | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        await asyncio.sleep(0.1)
        return {"step": simulation_step}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_turns_concurrent_scaling() -> None:
    num_agents = 25
    agents = [DummyAgent(str(i)) for i in range(num_agents)]
    sim = Simulation(agents=agents)

    start = time.perf_counter()
    results = await sim.run_turns_concurrent(agents)
    elapsed = time.perf_counter() - start
    throughput = len(results) / elapsed

    assert len(results) == num_agents
    assert sim.current_step == num_agents
    assert throughput > 5
