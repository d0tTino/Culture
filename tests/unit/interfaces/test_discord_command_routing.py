from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.interfaces import discord_bot
from src.interfaces.discord_bot import routing


@pytest.mark.unit
def test_help_text_snapshot() -> None:
    snapshot = Path("tests/unit/interfaces/snapshots/discord_help.txt").read_text().strip()
    assert discord_bot.build_help_text() == snapshot


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ephemeral_response_snapshot_for_help() -> None:
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock()),
        user=SimpleNamespace(id="user-1"),
        channel=SimpleNamespace(id=100),
    )

    await discord_bot.slash_help.callback(interaction)

    content = interaction.response.send_message.await_args.args[0]
    snapshot = Path("tests/unit/interfaces/snapshots/discord_help.txt").read_text().strip()
    assert content == snapshot
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True


class RecordingTree:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def command(self, name: str):
        def _decorator(fn):
            self.handlers[name] = fn
            return fn

        return _decorator


@pytest.mark.unit
@pytest.mark.asyncio
async def test_public_command_flow_uses_policy_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = RecordingTree()
    monkeypatch.setattr(routing, "POLICY_GATEWAY", SimpleNamespace(enforce=AsyncMock(return_value=True)))
    called = AsyncMock()

    async def _help(interaction):
        await called(interaction)

    monkeypatch.setattr(discord_bot, "slash_help", _help)

    routing.register_slash_commands(tree)

    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    await tree.handlers["help"](interaction)

    routing.POLICY_GATEWAY.enforce.assert_awaited_once_with(interaction, "help", admin_only=False)
    called.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_admin_command_flow_uses_policy_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = RecordingTree()
    monkeypatch.setattr(routing, "POLICY_GATEWAY", SimpleNamespace(enforce=AsyncMock(return_value=True)))
    called = AsyncMock()

    async def _start(interaction):
        await called(interaction)

    monkeypatch.setattr(discord_bot, "slash_start", _start)

    routing.register_slash_commands(tree)

    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    await tree.handlers["start"](interaction)

    routing.POLICY_GATEWAY.enforce.assert_awaited_once_with(interaction, "start", admin_only=True)
    called.assert_awaited_once()
