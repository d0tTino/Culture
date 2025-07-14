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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: asyncio.Queue[db.SimulationEvent | None] = asyncio.Queue()
    monkeypatch.setattr(db, "get_event_queue", lambda: queue)
    import src.sim.simulation as sim_mod

    monkeypatch.setattr(sim_mod, "get_event_queue", lambda: queue)
    from src.interfaces import discord_bot as bot

    monkeypatch.setattr(bot, "event_queue", queue)
    monkeypatch.setattr(bot, "get_event_queue", lambda: queue)

    agent = DummyAgent("A")
    sim = Simulation([agent])
    await sim.start_event_listener()

    await bot.slash_pause.callback(DummyInteraction())
    await asyncio.sleep(0.05)
    assert sim.paused is True

    await bot.slash_resume.callback(DummyInteraction())
    await asyncio.sleep(0.05)
    assert sim.paused is False

    await bot.slash_set_speed.callback(DummyInteraction(), value=2.5)
    await asyncio.sleep(0.05)
    assert sim.speed == pytest.approx(2.5)

    sim.close()
    await queue.put(None)
