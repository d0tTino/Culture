import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.infra import config
from src.sim.simulation import Simulation


class DummyState(SimpleNamespace):
    ip: float = 10.0
    du: float = 5.0
    age: int = 0
    is_alive: bool = True
    inheritance: float = 0.0
    genes: ClassVar[dict[str, float]] = {}
    parent_id: str | None = None
    short_term_memory: ClassVar[list] = []
    messages_sent_count: int = 0
    last_message_step: int = 0
    relationships: ClassVar[dict] = {}
    current_role: str = "dummy"
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
        return {}


@pytest.mark.integration
def test_agent_retirement_and_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_AGENT_AGE", "1")
    monkeypatch.setitem(config._CONFIG, "MAX_AGENT_AGE", 1)
    agent = DummyAgent("a1")
    sim = Simulation(agents=[agent])

    asyncio.get_event_loop().run_until_complete(sim.run_step())

    assert not agent.state.is_alive
    assert agent.state.inheritance == 15.0

    child = DummyAgent("a2")
    sim.spawn_agent(child, inheritance=agent.state.inheritance)
    assert child.state.ip == 25.0  # inherited 15 + default 10
    assert len(sim.agents) == 2
