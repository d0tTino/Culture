import asyncio
from types import SimpleNamespace
from typing import Callable
from unittest.mock import AsyncMock

import pytest

from src.interfaces import discord_moderation


class DummyContext:
    def __init__(self) -> None:
        self.queue = asyncio.Queue()

    def get_event_queue(self) -> asyncio.Queue:
        return self.queue


class DummyBot:
    def __init__(self, context: DummyContext) -> None:
        self.context = context


class DummyInteraction:
    def __init__(self) -> None:
        self.user = SimpleNamespace(id=123)
        self.response = SimpleNamespace(send_message=AsyncMock())


@pytest.fixture()
def bot(monkeypatch: pytest.MonkeyPatch) -> DummyBot:
    ctx = DummyContext()
    dummy = DummyBot(ctx)
    monkeypatch.setattr(discord_moderation, "get_active_bot", lambda: dummy)
    return dummy


@pytest.fixture()
def interaction() -> DummyInteraction:
    return DummyInteraction()


@pytest.fixture()
def interaction_factory() -> "Callable[[], DummyInteraction]":
    def factory() -> DummyInteraction:
        return DummyInteraction()

    return factory


@pytest.fixture(autouse=True)
def reset_rate_limit() -> None:
    discord_moderation._ACTION_COUNTS.clear()
    discord_moderation._COOLDOWNS.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limit_includes_agent_id(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_eval(content: str) -> tuple[bool, str]:
        calls.append(content)
        return True, content

    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", fake_eval)

    user = SimpleNamespace(id=123)
    allowed = await discord_moderation._rate_limit(user, "mute", "agentX")
    assert allowed is True
    assert calls == ["123:mute:agentX"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_mute_per_agent(
    monkeypatch: pytest.MonkeyPatch,
    bot: DummyBot,
    interaction_factory: Callable[[], DummyInteraction],
) -> None:
    async def fake_eval(content: str) -> tuple[bool, str]:
        if content == "123:mute:agent1":
            return False, ""
        return True, ""

    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", fake_eval)

    interaction1 = interaction_factory()
    await discord_moderation.slash_mute.callback(interaction1, "agent1")
    interaction1.response.send_message.assert_awaited_once_with("rate limited", ephemeral=True)

    interaction2 = interaction_factory()
    await discord_moderation.slash_mute.callback(interaction2, "agent2")
    interaction2.response.send_message.assert_awaited_once_with("muted", ephemeral=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_mute_flow(
    monkeypatch: pytest.MonkeyPatch, bot: DummyBot, interaction: DummyInteraction
) -> None:
    async def allow_eval(content: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", allow_eval)
    await discord_moderation.slash_mute.callback(interaction, "agent1")
    interaction.response.send_message.assert_awaited_once_with("muted", ephemeral=True)
    event = await bot.context.get_event_queue().get()
    assert event.type == "moderation"
    assert event.data == {"command": "mute", "agent_id": "agent1"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_penalty_flow(
    monkeypatch: pytest.MonkeyPatch, bot: DummyBot, interaction: DummyInteraction
) -> None:
    async def allow_eval(content: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", allow_eval)
    await discord_moderation.slash_penalty.callback(interaction, "agent1", 1.0, 2.0)
    interaction.response.send_message.assert_awaited_once_with("penalty applied", ephemeral=True)
    event = await bot.context.get_event_queue().get()
    assert event.type == "moderation"
    assert event.data == {
        "command": "penalty",
        "agent_id": "agent1",
        "ip": 1.0,
        "du": 2.0,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_reset_memory_flow(
    monkeypatch: pytest.MonkeyPatch, bot: DummyBot, interaction: DummyInteraction
) -> None:
    async def allow_eval(content: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", allow_eval)
    await discord_moderation.slash_reset_memory.callback(interaction, "agent1")
    interaction.response.send_message.assert_awaited_once_with("memory reset", ephemeral=True)
    event = await bot.context.get_event_queue().get()
    assert event.type == "moderation"
    assert event.data == {"command": "reset_memory", "agent_id": "agent1"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_unmute_flow(
    monkeypatch: pytest.MonkeyPatch, bot: DummyBot, interaction: DummyInteraction
) -> None:
    async def allow_eval(content: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", allow_eval)
    await discord_moderation.slash_unmute.callback(interaction, "agent1")
    interaction.response.send_message.assert_awaited_once_with("unmuted", ephemeral=True)
    event = await bot.context.get_event_queue().get()
    assert event.type == "moderation"
    assert event.data == {"command": "unmute", "agent_id": "agent1"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limit_counts_and_violation(
    monkeypatch: pytest.MonkeyPatch,
    bot: DummyBot,
    interaction_factory: Callable[[], DummyInteraction],
) -> None:
    discord_moderation._ACTION_COUNTS.clear()
    discord_moderation._COOLDOWNS.clear()
    monkeypatch.setattr(discord_moderation, "_COOLDOWN_SECONDS", 999.0)

    async def allow_eval(content: str) -> tuple[bool, str]:
        return True, ""

    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", allow_eval)

    interaction1 = interaction_factory()
    await discord_moderation.slash_mute.callback(interaction1, "agent1")
    await bot.context.get_event_queue().get()
    assert discord_moderation._ACTION_COUNTS["123:mute"] == 1

    interaction2 = interaction_factory()
    await discord_moderation.slash_mute.callback(interaction2, "agent1")
    interaction2.response.send_message.assert_awaited_once_with("rate limited", ephemeral=True)
    event = await bot.context.get_event_queue().get()
    assert event.data == {"command": "mute", "agent_id": "agent1", "violation": True}
    assert discord_moderation._ACTION_COUNTS["123:mute"] == 2
