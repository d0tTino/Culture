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

    handled: list[str] = []
    original = Simulation._handle_human_command

    async def wrapped(self: Simulation, text: str) -> None:
        handled.append(text)
        await original(self, text)

    monkeypatch.setattr(Simulation, "_handle_human_command", wrapped)

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
