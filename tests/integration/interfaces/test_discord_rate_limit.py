from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

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
async def test_mute_unmute_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
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
    mock_reward = MagicMock()
    monkeypatch.setattr(discord_moderation, "log_reward", mock_reward)
    interaction = DummyInteraction()

    await discord_moderation.slash_mute.callback(interaction, "agent1")
    await discord_moderation.slash_mute.callback(interaction, "agent1")
    await discord_moderation.slash_unmute.callback(interaction, "agent1")

    assert interaction.response.send_message.await_args_list == [
        call("muted", ephemeral=True),
        call("rate limited", ephemeral=True),
        call("unmuted", ephemeral=True),
    ]

    events = [await queue.get() for _ in range(4)]
    assert events[0].data == {"command": "mute", "agent_id": "agent1"}
    assert events[1].data == {"command": "mute", "agent_id": "agent1", "violation": True}
    assert events[2].data == {
        "command": "penalty",
        "agent_id": "agent1",
        "ip": discord_moderation._IP_PENALTY,
        "du": discord_moderation._DU_PENALTY,
    }
    assert events[3].data == {"command": "unmute", "agent_id": "agent1"}
    mock_reward.assert_called_once_with(
        "agent1",
        -discord_moderation._IP_PENALTY,
        -discord_moderation._DU_PENALTY,
        "rate_limit_violation",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reset_memory_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
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
    mock_reward = MagicMock()
    monkeypatch.setattr(discord_moderation, "log_reward", mock_reward)
    interaction = DummyInteraction()

    await discord_moderation.slash_reset_memory.callback(interaction, "agent1")
    await discord_moderation.slash_reset_memory.callback(interaction, "agent1")

    assert interaction.response.send_message.await_args_list == [
        call("memory reset", ephemeral=True),
        call("rate limited", ephemeral=True),
    ]

    events = [await queue.get() for _ in range(3)]
    assert events[0].data == {"command": "reset_memory", "agent_id": "agent1"}
    assert events[1].data == {
        "command": "reset_memory",
        "agent_id": "agent1",
        "violation": True,
    }
    assert events[2].data == {
        "command": "penalty",
        "agent_id": "agent1",
        "ip": discord_moderation._IP_PENALTY,
        "du": discord_moderation._DU_PENALTY,
    }
    mock_reward.assert_called_once_with(
        "agent1",
        -discord_moderation._IP_PENALTY,
        -discord_moderation._DU_PENALTY,
        "rate_limit_violation",
    )
