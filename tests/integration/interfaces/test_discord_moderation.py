from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

pytest.importorskip("discord")

from src.interfaces import discord_moderation
from src.sim.context import SimulationContext


class DummyInteraction:
    def __init__(self) -> None:
        self.user = SimpleNamespace(id=123)
        self.response = SimpleNamespace(send_message=AsyncMock())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_mute_calls_penalty_event(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SimulationContext()
    queue = ctx.get_event_queue()
    dummy_bot = SimpleNamespace(context=ctx)
    monkeypatch.setattr(discord_moderation, "get_active_bot", lambda: dummy_bot)
    monkeypatch.setattr(
        discord_moderation, "evaluate_with_opa", AsyncMock(return_value=(True, ""))
    )
    monkeypatch.setattr(discord_moderation.time, "monotonic", lambda: 0.0)
    discord_moderation._ACTION_COUNTS.clear()
    discord_moderation._COOLDOWNS.clear()
    interaction = DummyInteraction()

    await discord_moderation.slash_mute.callback(interaction, "agent1")
    await discord_moderation.slash_mute.callback(interaction, "agent1")

    assert interaction.response.send_message.await_args_list == [
        call("muted", ephemeral=True),
        call("rate limited", ephemeral=True),
    ]

    events = [await queue.get() for _ in range(3)]
    assert events[0].data == {"command": "mute", "agent_id": "agent1"}
    assert events[1].data == {"command": "mute", "agent_id": "agent1", "violation": True}
    assert events[2].data == {
        "command": "penalty",
        "agent_id": "agent1",
        "ip": discord_moderation._IP_PENALTY,
        "du": discord_moderation._DU_PENALTY,
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mute_denied_by_opa_triggers_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SimulationContext()
    queue = ctx.get_event_queue()
    dummy_bot = SimpleNamespace(context=ctx)
    monkeypatch.setattr(discord_moderation, "get_active_bot", lambda: dummy_bot)
    eval_mock = AsyncMock(side_effect=[(True, ""), (False, "")])
    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", eval_mock)
    times = iter([0.0, 2.0])
    monkeypatch.setattr(discord_moderation.time, "monotonic", lambda: next(times))
    discord_moderation._ACTION_COUNTS.clear()
    discord_moderation._COOLDOWNS.clear()
    interaction = DummyInteraction()

    await discord_moderation.slash_mute.callback(interaction, "agent1")
    await discord_moderation.slash_mute.callback(interaction, "agent1")

    assert interaction.response.send_message.await_args_list == [
        call("muted", ephemeral=True),
        call("rate limited", ephemeral=True),
    ]

    events = [await queue.get() for _ in range(3)]
    assert events[0].data == {"command": "mute", "agent_id": "agent1"}
    assert events[1].data == {"command": "mute", "agent_id": "agent1", "violation": True}
    assert events[2].data == {
        "command": "penalty",
        "agent_id": "agent1",
        "ip": discord_moderation._IP_PENALTY,
        "du": discord_moderation._DU_PENALTY,
    }
    assert eval_mock.await_args_list == [
        call("123:mute:agent1"),
        call("123:mute:agent1"),
    ]
