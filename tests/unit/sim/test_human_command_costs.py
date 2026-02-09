import asyncio
import sys
import types
from pathlib import Path

import pytest

from src.infra import config

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
        self.genes = {}


class DummyAgent:
    def __init__(self, agent_id: str, ip: float = 2.0, du: float = 2.0) -> None:
        self.agent_id = agent_id
        self._state = DummyState(ip, du)
        from src.infra.ledger import ledger as _ledger

        _ledger.log_change(agent_id, ip, du, "init")

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyState:  # pragma: no cover - simple accessor
        return self._state

    def update_state(self, state: DummyState) -> None:  # pragma: no cover - unused
        self._state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:  # pragma: no cover - unused
        return {}


@pytest.mark.asyncio
async def test_handle_human_command_missing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())

    dashboard_stub = types.ModuleType("src.interfaces.dashboard_backend")

    class _SimulationEvent:  # pragma: no cover - simple placeholder
        pass

    def _emit_event(event: _SimulationEvent) -> None:  # pragma: no cover - placeholder
        return None

    def _emit_map_action_event(*args: object, **kwargs: object) -> None:  # pragma: no cover
        return None

    dashboard_stub.SimulationEvent = _SimulationEvent
    dashboard_stub.emit_event = _emit_event
    dashboard_stub.emit_map_action_event = _emit_map_action_event
    dashboard_stub.get_event_queue = lambda: asyncio.Queue()
    monkeypatch.setitem(sys.modules, "src.interfaces.dashboard_backend", dashboard_stub)

    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    agent = DummyAgent("A")
    sim = Simulation([agent])

    for key in [
        "IP_COST_SEND_DIRECT_MESSAGE",
        "IP_COST_BROADCAST_MESSAGE",
        "DU_COST_PER_ACTION",
        "DU_COST_BROADCAST_ACTION",
    ]:
        monkeypatch.setitem(config._CONFIG, key, None)

    await sim._handle_human_command("hello")
    sim._last_relay_time = 0.0
    await sim._handle_human_command("/broadcast hi")

    assert agent.state.ip == pytest.approx(2.0)
    assert agent.state.du == pytest.approx(2.0)
    async with sim._msg_lock:
        assert len(sim.pending_messages_for_next_round) == 2
    sim.close()


@pytest.mark.asyncio
async def test_human_command_uses_human_budget_without_agent_state_deduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)
    monkeypatch.setitem(config._CONFIG, "HUMAN_COMMAND_BUDGET_AGENT_ID", "human")

    agent = DummyAgent("A")
    ledger.log_change("human", 10.0, 10.0, "init")
    sim = Simulation([agent])

    monkeypatch.setattr(
        "src.sim.simulation.get_resource_manager",
        lambda: types.SimpleNamespace(ensure_du_budget=lambda *_args, **_kwargs: None),
    )

    await sim._handle_human_command("hello")

    assert agent.state.ip == pytest.approx(2.0)
    hip, hdu = ledger.get_balance("human")
    assert hip == pytest.approx(9.0)
    assert hdu == pytest.approx(9.0)
    sim.close()
