from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.interfaces import discord_bot

EXPECTED_HELP_TEXT = "\n".join(
    [
        "## 👋 Culture Bot Help",
        "### Public commands",
        "- `/dm <agent_id> <message>` — send a direct message to one agent.",
        "- `/broadcast <message>` — send a message to all agents.",
        "- `/kb <text>` — add a note to the Knowledge Board.",
        "- `/status`, `/stats` — view current state/metrics.",
        "- `/start_here` — show onboarding, modes, and scenario cards.",
        "- `/propose`, `/propose_law`, `/vote` — governance interactions.",
        "",
        "### Moderator/Admin commands",
        "- `/nudge <prompt>` — steer agent behavior *(moderator/admin)*.",
        "- `/event <text>` — inject world events *(moderator/admin)*.",
        "- `/start`, `/stop`, `/pause`, `/resume` — sim lifecycle *(moderator/admin)*.",
        "- `/spawn`, `/kill_agent`, `/pause_all`, `/kill` — high-impact controls *(admin required for kill/pause_all/kill_agent)*.",
        "- `/set_speed`, `/speed`, `/set_max_rate` — tuning controls *(admin required for set_max_rate)*.",
        "",
        "### Quick examples",
        "- `/dm agent-2 What's your latest plan?`",
        "- `/broadcast Team sync in 2 minutes.`",
        "- `/kb Rule: cite data source before proposing policy.`",
        "- `/nudge Consider long-term coalition outcomes.`",
        "",
        "### Permission + rate-limit notes",
        "- Public commands are usable by all channel users unless noted.",
        "- Admin-only commands require Discord administrator privileges.",
        "- Slash commands are globally rate-limited per user (default: 5 commands / 60s).",
        "- Some moderation actions also have cooldowns to reduce spam.",
        "",
        "### User modes",
        "- `observer` — read-only guidance and context.",
        "- `participant` — regular conversation with agents.",
        "- `world-shaper` — propose world-level interventions.",
        "- `moderator` — policy and safety operations.",
    ]
)

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
