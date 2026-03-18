import sys
from unittest.mock import AsyncMock

import pytest

from src.governance.decision_kernel import DecisionProvenance
from src.interfaces import dashboard_backend as db
from src.interfaces.dashboard_backend import SimulationEvent
from src.interfaces.interaction_schema import InteractionResult

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

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        return {}


@pytest.mark.asyncio
async def test_legacy_human_ingestion_routes_through_command_bus() -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    sim = Simulation([DummyAgent("A")])
    sim.command_bus.dispatch_payload = AsyncMock()

    await sim.external_event_ingestion.handle_human_command("hello", {"sender_id": "user-1"})

    sim.command_bus.dispatch_payload.assert_awaited_once()
    await sim.stop_event_listener()
    sim.close()


@pytest.mark.asyncio
async def test_event_bus_control_routes_through_command_bus() -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    sim = Simulation([DummyAgent("A")])
    sim.command_bus.dispatch_payload = AsyncMock()

    evt = SimulationEvent(type="control", data={"command": "pause"})
    await sim.external_event_ingestion.route_event(evt)

    sim.command_bus.dispatch_payload.assert_awaited_once()
    await sim.stop_event_listener()
    sim.close()


@pytest.mark.asyncio
async def test_dashboard_control_routes_through_dispatcher() -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    sim = Simulation([DummyAgent("A")])
    dispatched = AsyncMock(
        return_value=InteractionResult(
            status="ok",
            user_message="Command accepted.",
            reason_code="ok",
            data={"paused": True},
            decision_provenance=DecisionProvenance(policy_id="p", rule_id="r"),
        )
    )
    sim.command_dispatcher.dispatch_payload = dispatched

    state = dict(db.DEFAULT_CONTEXT.sim_state)
    state["simulation"] = sim
    ctx = db.SimulationContext(sim_state=state)

    result = await db.handle_control_command({"command": "pause"}, ctx=ctx)

    dispatched.assert_awaited_once()
    assert result["status"] == "ok"
    await sim.stop_event_listener()
    sim.close()


@pytest.mark.asyncio
async def test_run_step_routes_through_progress_method() -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.sim.simulation import Simulation

    sim = Simulation([DummyAgent("A")])
    sim._progress_step = AsyncMock(return_value=7)

    out = await sim.run_step(max_turns=3)

    assert out == 7
    sim._progress_step.assert_awaited_once_with(max_turns=3)
    await sim.stop_event_listener()
    sim.close()
