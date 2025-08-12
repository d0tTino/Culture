import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.interfaces import discord_moderation


class DummyContext:
    def __init__(self) -> None:
        self.queue = asyncio.Queue()

    def get_event_queue(self) -> asyncio.Queue:
        return self.queue


class DummyBot:
    def __init__(self) -> None:
        self.context = DummyContext()


class DummyInteraction:
    def __init__(self) -> None:
        self.user = SimpleNamespace(id=123)
        self.response = SimpleNamespace(send_message=AsyncMock())


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
async def test_slash_mute_per_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_eval(content: str) -> tuple[bool, str]:
        if content == "123:mute:agent1":
            return False, ""
        return True, ""

    monkeypatch.setattr(discord_moderation, "evaluate_with_opa", fake_eval)
    monkeypatch.setattr(discord_moderation, "get_active_bot", lambda: DummyBot())

    interaction1 = DummyInteraction()
    await discord_moderation.slash_mute.callback(interaction1, "agent1")
    interaction1.response.send_message.assert_awaited_once_with(
        "rate limited", ephemeral=True
    )

    interaction2 = DummyInteraction()
    await discord_moderation.slash_mute.callback(interaction2, "agent2")
    interaction2.response.send_message.assert_awaited_once_with(
        "muted", ephemeral=True
    )
