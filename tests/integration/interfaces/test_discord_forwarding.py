import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.interfaces import dashboard_backend as db
from src.sim import event_bus
from src.sim.simulation import Simulation


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_event_bus_forwarding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    bus = event_bus.EventBus()
    monkeypatch.setattr(event_bus, "get_event_bus", lambda: bus)
    monkeypatch.setattr("src.sim.context.get_event_bus", lambda: bus)
    monkeypatch.setattr("src.sim.event_kernel.get_event_bus", lambda: bus)
    monkeypatch.setattr("src.sim.simulation.get_event_bus", lambda: bus)

    from src.sim.runtime.external_event_ingestion_service import ExternalEventIngestionService

    handled: list[str] = []
    original = ExternalEventIngestionService.handle_human_command

    async def wrapped(
        self: ExternalEventIngestionService, text: str, metadata: dict | None = None
    ) -> None:
        handled.append(text)
        await original(self, text, metadata)

    monkeypatch.setattr(ExternalEventIngestionService, "handle_human_command", wrapped)

    with patch("src.interfaces.discord_bot.SimulationDiscordBot", AsyncMock()):
        agent = DummyAgent("A")
        sim = Simulation([agent])
        await sim.start_event_listener()
        await asyncio.sleep(0)

        await bus.publish(
            db.SimulationEvent(
                type="broadcast", data={"author": "human", "content": "/broadcast hello"}
            )
        )
        await asyncio.sleep(0.1)

        assert handled == ["/broadcast hello"]
        assert agent.state.ip == pytest.approx(1.0)
        assert agent.state.du == pytest.approx(1.0)

        sim.close()
        bus.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_human_command_rate_limit_scoped_by_sender(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    import src.sim.simulation as simulation_module
    from src.infra import ledger as ledger_module
    from src.infra.ledger import Ledger

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)
    monkeypatch.setattr(
        simulation_module,
        "get_resource_manager",
        lambda: type(
            "ResourceManager",
            (),
            {
                "ensure_du_budget": staticmethod(lambda *_a, **_k: None),
                "set_du_budget": staticmethod(lambda *_a, **_k: None),
                "cap_tick": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )
    spend_mock = AsyncMock()
    monkeypatch.setattr(ledger_module.ledger, "spend", spend_mock)
    emit_mock = AsyncMock()
    monkeypatch.setattr(simulation_module, "emit_event", emit_mock)

    sim = Simulation([DummyAgent("A")])

    await sim.external_event_ingestion.handle_human_command(
        "hello from user 1", {"sender_id": "user-1"}
    )
    await sim.external_event_ingestion.handle_human_command(
        "hello from user 2", {"sender_id": "user-2"}
    )

    assert spend_mock.await_count == 2
    assert emit_mock.await_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_human_command_rate_limit_sends_feedback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    import src.sim.simulation as simulation_module
    from src.infra import ledger as ledger_module
    from src.infra.ledger import Ledger

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)
    monkeypatch.setattr(
        simulation_module,
        "get_resource_manager",
        lambda: type(
            "ResourceManager",
            (),
            {
                "ensure_du_budget": staticmethod(lambda *_a, **_k: None),
                "set_du_budget": staticmethod(lambda *_a, **_k: None),
                "cap_tick": staticmethod(lambda *_a, **_k: None),
            },
        )(),
    )
    spend_mock = AsyncMock()
    monkeypatch.setattr(ledger_module.ledger, "spend", spend_mock)
    emit_mock = AsyncMock()
    monkeypatch.setattr(simulation_module, "emit_event", emit_mock)

    sim = Simulation([DummyAgent("A")])

    await sim.external_event_ingestion.handle_human_command("first", {"sender_id": "user-1"})
    await sim.external_event_ingestion.handle_human_command("second", {"sender_id": "user-1"})

    assert spend_mock.await_count == 1
    assert emit_mock.await_count == 1
    throttled_event = emit_mock.await_args.args[0]
    assert throttled_event.type == "human_command_rate_limited"
    assert throttled_event.data["sender_id"] == "user-1"
