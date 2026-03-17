from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.interfaces import discord_bot

EXPECTED_HELP_TEXT = Path("tests/unit/interfaces/snapshots/discord_help.txt").read_text().strip()

def test_build_help_text_is_stable() -> None:
    assert discord_bot.build_help_text() == EXPECTED_HELP_TEXT


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_help_returns_help_text() -> None:
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock()),
        user=SimpleNamespace(id="user-1"),
        channel=SimpleNamespace(id=123),
    )

    await discord_bot.slash_help.callback(interaction)

    interaction.response.send_message.assert_awaited_once_with(
        EXPECTED_HELP_TEXT,
        ephemeral=True,
    )


@pytest.mark.unit
def test_create_onboarding_embed_contains_interaction_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    fields: list[tuple[str, str, bool]] = []

    class DummyEmbed:
        def __init__(self, title: str, description: str, color: int) -> None:
            self.title = title
            self.description = description
            self.color = color
            self.footer_text = ""

        def add_field(self, name: str, value: str, inline: bool) -> None:
            fields.append((name, value, inline))

        def set_footer(self, text: str) -> None:
            self.footer_text = text

    monkeypatch.setattr(
        discord_bot,
        "discord",
        SimpleNamespace(Embed=DummyEmbed, Color=SimpleNamespace(blue=lambda: 123)),
    )

    embed = discord_bot.create_onboarding_embed(99)

    assert embed.title == "🧭 How to interact"
    assert "slash commands" in embed.description
    assert any(name == "Permissions" for name, _, _ in fields)
    assert any(name == "Modes" for name, _, _ in fields)
    assert any(name == "Rate limits" for name, _, _ in fields)
    assert embed.footer_text == "Channel ID: 99"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_start_here_returns_scenarios() -> None:
    interaction = SimpleNamespace(
        response=SimpleNamespace(send_message=AsyncMock()),
        user=SimpleNamespace(id="user-1"),
        channel=SimpleNamespace(id=123),
    )

    await discord_bot.slash_start_here.callback(interaction)

    sent = interaction.response.send_message.await_args.args[0]
    assert "Start Here" in sent
    assert "observer" in sent
    assert "Scenario cards" in sent
