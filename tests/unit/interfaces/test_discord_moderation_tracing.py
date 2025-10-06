"""Tests for Discord moderation tracing spans."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.interfaces import discord_moderation


class _FakeContext:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[object] = asyncio.Queue()

    def get_event_queue(self) -> asyncio.Queue[object]:
        return self.queue


@pytest.fixture
def interaction() -> SimpleNamespace:
    return SimpleNamespace(
        user=SimpleNamespace(id="user-1"),
        channel=SimpleNamespace(id="channel-9"),
        response=SimpleNamespace(send_message=AsyncMock()),
    )


@pytest.fixture
def patched_bot() -> asyncio.Queue[object]:
    ctx = _FakeContext()
    bot = SimpleNamespace(context=ctx)
    with patch("src.interfaces.discord_moderation.get_active_bot", return_value=bot):
        yield ctx.queue


@pytest.fixture
def command_span_mock() -> MagicMock:
    with patch("src.interfaces.discord_moderation.discord_bot.command_span") as mock:
        mock.side_effect = lambda *args, **kwargs: contextlib.nullcontext()
        yield mock


@pytest.mark.asyncio
async def test_slash_mute_uses_command_span(
    interaction: SimpleNamespace,
    patched_bot: asyncio.Queue[object],
    command_span_mock: MagicMock,
) -> None:
    rate_limit = AsyncMock(return_value=True)
    with patch("src.interfaces.discord_moderation._rate_limit", new=rate_limit):
        await discord_moderation.slash_mute(interaction, "agent-42")

    command_span_mock.assert_called_once_with("mute", interaction, agent_id="agent-42")
    rate_limit.assert_awaited_once()


@pytest.mark.asyncio
async def test_slash_reset_memory_uses_command_span(
    interaction: SimpleNamespace,
    patched_bot: asyncio.Queue[object],
    command_span_mock: MagicMock,
) -> None:
    rate_limit = AsyncMock(return_value=True)
    with (
        patch("src.interfaces.discord_moderation._rate_limit", new=rate_limit),
        patch("src.interfaces.discord_moderation.has_admin_permission", return_value=True),
    ):
        await discord_moderation.slash_reset_memory(interaction, "agent-77")

    command_span_mock.assert_called_once_with(
        "reset_memory", interaction, agent_id="agent-77"
    )
    rate_limit.assert_awaited_once()


@pytest.mark.asyncio
async def test_slash_penalty_uses_command_span(
    interaction: SimpleNamespace,
    patched_bot: asyncio.Queue[object],
    command_span_mock: MagicMock,
) -> None:
    rate_limit = AsyncMock(return_value=True)
    with patch("src.interfaces.discord_moderation._rate_limit", new=rate_limit):
        await discord_moderation.slash_penalty(interaction, "agent-5", ip=1.5, du=0.25)

    command_span_mock.assert_called_once_with(
        "penalty", interaction, agent_id="agent-5"
    )
    rate_limit.assert_awaited_once()


@pytest.mark.asyncio
async def test_slash_unmute_traces_rate_limited_paths(
    interaction: SimpleNamespace,
    patched_bot: asyncio.Queue[object],
    command_span_mock: MagicMock,
) -> None:
    rate_limit = AsyncMock(return_value=False)
    interaction.response.send_message.reset_mock()
    with (
        patch("src.interfaces.discord_moderation._rate_limit", new=rate_limit),
        patch("src.interfaces.discord_moderation.log_penalty"),
    ):
        await discord_moderation.slash_unmute(interaction, "agent-13")

    command_span_mock.assert_called_once_with("unmute", interaction, agent_id="agent-13")
    rate_limit.assert_awaited_once()
    interaction.response.send_message.assert_awaited_once_with(
        "rate limited", ephemeral=True
    )
