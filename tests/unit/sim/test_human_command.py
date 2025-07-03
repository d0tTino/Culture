import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class DummyState:
    def __init__(self, ip: float = 2.0, du: float = 2.0) -> None:
        self.ip = ip
        self.du = du
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str, ip: float = 2.0, du: float = 2.0) -> None:
        self.agent_id = agent_id
        self._state = DummyState(ip, du)
        from src.infra.ledger import ledger as _ledger

        _ledger.log_change(agent_id, ip, du, "init")

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyState:
        return self._state

    def update_state(self, state: DummyState) -> None:
        self._state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        return {}


@pytest.mark.asyncio
async def test_human_command_deducts_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    agent = DummyAgent("A")
    sim = Simulation([agent])

    await sim._handle_human_command("hello")

    assert agent.state.ip == pytest.approx(1.0)
    assert agent.state.du == pytest.approx(1.0)
    ip, du = ledger.get_balance(agent.agent_id)
    assert ip == pytest.approx(1.0)
    assert du == pytest.approx(1.0)
    async with sim._msg_lock:
        assert sim.pending_messages_for_next_round[-1]["content"] == "hello"


@pytest.mark.asyncio
async def test_human_command_rejected_without_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    agent = DummyAgent("A", ip=0.5, du=0.5)
    sim = Simulation([agent])

    await sim._handle_human_command("hello")

    assert agent.state.ip == pytest.approx(0.5)
    assert agent.state.du == pytest.approx(0.5)
    ip, du = ledger.get_balance(agent.agent_id)
    assert ip == pytest.approx(0.5)
    assert du == pytest.approx(0.5)
    async with sim._msg_lock:
        assert sim.pending_messages_for_next_round == []
