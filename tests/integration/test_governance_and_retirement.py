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

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict:
        return {}

    def update_state(self, new_state: DummyState) -> None:
        self.state = new_state


@pytest.mark.asyncio
@pytest.mark.integration
async def test_law_proposal() -> None:
    agents = [DummyAgent("a1"), DummyAgent("a2")]
    sim = Simulation(agents=agents)
    approved = await sim.propose_law("a1", "no spamming")
    assert approved
    assert any("Law approved" in e["content_display"] for e in sim.knowledge_board.entries)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_retirement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(config._CONFIG, "MAX_AGENT_AGE", 2)
    agent = DummyAgent("a1")
    sim = Simulation(agents=[agent])

    await sim.run_step()
    assert agent.state.is_alive
    await sim.run_step()
    assert not agent.state.is_alive
    assert agent.state.inheritance > 0
