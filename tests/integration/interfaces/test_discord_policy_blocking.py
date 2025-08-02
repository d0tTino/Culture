import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.interfaces.discord_bot import SimulationDiscordBot
from src.sim import event_bus
from src.sim.context import SimulationContext
from src.sim.simulation import Simulation


class DummyDiscordClient:
    """Simple stand-in for ``discord.Client`` that records events."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._events: dict[str, object] = {}

        def _event(func: object) -> object:
            if hasattr(func, "__name__"):
                self._events[func.__name__] = func
            return func

        self.event = _event
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> object:  # pragma: no cover - minimal
        class DummyChannel:
            async def send(self_inner, *args: object, **kwargs: object) -> None:
                pass

        return DummyChannel()


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
async def test_blocked_messages_do_not_reach_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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

    ctx = SimulationContext()
    monkeypatch.setattr("src.interfaces.dashboard_backend.DEFAULT_CONTEXT", ctx)
    monkeypatch.setattr("src.interfaces.dashboard_backend.SIM_STATE", ctx.sim_state)

    monkeypatch.setattr("src.interfaces.discord_bot.allow_message", lambda _: True)
    monkeypatch.setattr(
        "src.interfaces.discord_bot.evaluate_with_opa",
        AsyncMock(return_value=(False, "blocked")),
    )

    handled: list[str] = []

    async def handler(self: Simulation, text: str) -> None:
        handled.append(text)

    monkeypatch.setattr(Simulation, "_handle_human_command", handler)

    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        bot = await SimulationDiscordBot.create("token", 123, context=ctx)

    agent = DummyAgent("A")
    sim = Simulation([agent])
    await sim.start_event_listener()

    on_msg = bot.client._events["on_message"]
    msg = MagicMock()
    msg.content = "hello"
    msg.author = MagicMock()
    msg.author.id = 1
    msg.channel = MagicMock()
    msg.channel.id = 2
    await on_msg(msg)
    await asyncio.sleep(0.1)

    assert handled == []

    sim.close()
    bus.shutdown()
