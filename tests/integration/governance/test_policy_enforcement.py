from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.agents.core.agent_controller import AgentController
from src.infra import config
from src.sim.simulation import Simulation


class DummyState(SimpleNamespace):
    ip: float = 0.0
    du: float = 0.0
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

    def get_id(self) -> str:
        return self.agent_id

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
async def test_vote_on_policy_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import governance
    from src.governance import policy as gpolicy

    async def deny(_: str) -> bool:
        return False

    monkeypatch.setattr(gpolicy, "evaluate_policy", deny)
    monkeypatch.setattr(governance, "evaluate_policy", deny)

    controller = AgentController()
    allowed = await controller.vote_on_policy("bad law")
    assert allowed is False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rejected_proposal_not_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    async def deny(_: str) -> tuple[bool, str]:
        return False, ""

    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")
    monkeypatch.setattr("src.utils.policy.evaluate_with_opa", deny)

    agents = [DummyAgent("a1"), DummyAgent("a2")]
    sim = Simulation(agents=agents)

    approved = await sim.propose_law("a1", "bad law")
    assert approved is False
    entries = [e["content_display"] for e in sim.knowledge_board.entries]
    assert any("Law proposed" in e for e in entries)
    assert not any("Law approved" in e for e in entries)
