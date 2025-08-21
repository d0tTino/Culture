import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.interfaces import dashboard_backend as db
from src.sim.simulation import Simulation


class DummyAgentState:
    def __init__(self) -> None:
        self.ip = 0.0
        self.du = 0.0
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = DummyAgentState()

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: DummyAgentState) -> None:
        self.state = state

    async def run_turn(self, *args: object, **kwargs: object) -> dict:
        return {}


class DummyInteraction:
    def __init__(self) -> None:
        self.response = SimpleNamespace(send_message=AsyncMock())


class DummyAdminInteraction(DummyInteraction):
    def __init__(self) -> None:
        super().__init__()
        self.user = SimpleNamespace(guild_permissions=SimpleNamespace(administrator=True))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: asyncio.Queue[db.SimulationEvent | None] = asyncio.Queue()

    class DummyBus:
        def __init__(self) -> None:
            self.queue = queue

        def subscribe(self) -> asyncio.Queue[db.SimulationEvent | None]:
            return self.queue

        def unsubscribe(self, q: asyncio.Queue[db.SimulationEvent | None]) -> None:
            return None

        async def publish(self, event: db.SimulationEvent | None) -> None:
            await self.queue.put(event)

        def shutdown(self) -> None:
            return None

    dummy_bus = DummyBus()

    import src.sim.context as ctx_mod
    import src.sim.event_bus as eb_mod
    import src.sim.simulation as sim_mod

    monkeypatch.setattr(eb_mod, "get_event_bus", lambda: dummy_bus)
    monkeypatch.setattr(ctx_mod, "get_event_bus", lambda: dummy_bus)
    monkeypatch.setattr(sim_mod, "get_event_bus", lambda: dummy_bus)

    from src.interfaces import discord_bot as bot

    agent = DummyAgent("A")
    sim = Simulation([agent])
    await sim.start_event_listener()

    await bot.slash_pause.callback(DummyInteraction())
    await asyncio.sleep(0.05)
    assert sim.paused is True

    await bot.slash_resume.callback(DummyInteraction())
    await asyncio.sleep(0.05)
    assert sim.paused is False

    await sim.handle_control_command({"command": "pause_all"})
    assert sim.paused is True

    await sim.handle_control_command({"command": "kill_agent", "agent_id": "A"})
    assert agent.state.is_alive is False

    await sim.handle_control_command({"command": "set_speed", "value": 2.5})
    assert sim.speed == pytest.approx(2.5)

    sim.close()
    dummy_bus.shutdown()
