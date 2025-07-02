import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.infra import config

sys.modules.setdefault("neo4j", SimpleNamespace(Driver=object, GraphDatabase=object))

from src.sim.simulation import Simulation


class DummyState(SimpleNamespace):
    ip: float = 1.0
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
    def __init__(self, agent_id: str, ip: float = 1.0) -> None:
        self.agent_id = agent_id
        self.state = DummyState()
        self.state.ip = ip

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
async def test_law_passes_majority_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")
    monkeypatch.setitem(config._CONFIG, "LAW_PASS_IP_REWARD", 1)
    monkeypatch.setitem(config._CONFIG, "LAW_PASS_DU_REWARD", 0)
    monkeypatch.setattr(config, "LAW_PASS_IP_REWARD", 1, raising=False)
    monkeypatch.setattr(config, "LAW_PASS_DU_REWARD", 0, raising=False)

    calls: list[tuple] = []
    monkeypatch.setattr("src.infra.ledger.ledger.log_change", lambda *a, **k: calls.append(a))

    results = [
        (True, ""),  # proposal allowed
        (True, ""),  # agent1 yes
        (False, ""),  # agent2 no
        (True, ""),  # agent3 yes
    ]

    async def side_effect(_: str) -> tuple[bool, str]:
        return results.pop(0)

    monkeypatch.setattr("src.utils.policy.evaluate_with_opa", side_effect)

    agents = [DummyAgent("a1"), DummyAgent("a2"), DummyAgent("a3")]
    sim = Simulation(agents=agents)
    approved = await sim.propose_law("a1", "test law")
    assert approved is True
    entries = [e["content_display"] for e in sim.knowledge_board.entries]
    assert any("Law proposed" in e for e in entries)
    assert any("Law approved" in e for e in entries)
    assert calls and calls[0][0] == "a1"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_law_fails_majority_no(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")
    results = [
        (True, ""),
        (False, ""),
        (False, ""),
        (True, ""),
    ]

    async def side_effect(_: str) -> tuple[bool, str]:
        return results.pop(0)

    monkeypatch.setattr("src.utils.policy.evaluate_with_opa", side_effect)
    calls: list[tuple] = []
    monkeypatch.setattr("src.infra.ledger.ledger.log_change", lambda *a, **k: calls.append(a))

    agents = [DummyAgent("a1"), DummyAgent("a2"), DummyAgent("a3")]
    sim = Simulation(agents=agents)
    approved = await sim.propose_law("a1", "test law")
    assert approved is False
    entries = [e["content_display"] for e in sim.knowledge_board.entries]
    assert any("Law proposed" in e for e in entries)
    assert not any("Law approved" in e for e in entries)
    assert not calls

@pytest.mark.asyncio
@pytest.mark.integration
async def test_quadratic_yes_overrides_majority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")

    async def allow(_: str) -> tuple[bool, str]:
        return True, ""

    votes = [True, False, False]

    async def vote_side_effect(_: object, __: str) -> bool:
        return votes.pop(0)

    monkeypatch.setattr("src.utils.policy.evaluate_with_opa", allow)
    monkeypatch.setattr("src.governance.voting._vote", vote_side_effect)

    agents = [
        DummyAgent("a1", ip=16),
        DummyAgent("a2", ip=1),
        DummyAgent("a3", ip=1),
    ]
    sim = Simulation(agents=agents)
    approved = await sim.propose_law("a1", "test law")
    assert approved is True


@pytest.mark.asyncio
@pytest.mark.integration
async def test_quadratic_no_overrides_majority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")

    async def allow(_: str) -> tuple[bool, str]:
        return True, ""

    votes = [True, True, False]

    async def vote_side_effect(_: object, __: str) -> bool:
        return votes.pop(0)

    monkeypatch.setattr("src.utils.policy.evaluate_with_opa", allow)
    monkeypatch.setattr("src.governance.voting._vote", vote_side_effect)

    agents = [
        DummyAgent("a1", ip=1),
        DummyAgent("a2", ip=1),
        DummyAgent("a3", ip=16),
    ]
    sim = Simulation(agents=agents)
    approved = await sim.propose_law("a1", "test law")
    assert approved is False
