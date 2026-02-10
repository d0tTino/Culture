from types import SimpleNamespace

import pytest

from src.interfaces import discord_bot


class DummyClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def advance(self, amount: float) -> None:
        self.value += amount


@pytest.mark.unit
@pytest.mark.asyncio
async def test_check_command_rate_limit_resets_after_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = DummyClock()
    monkeypatch.setattr(discord_bot.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(discord_bot, "_MAX_RATE", 2)
    monkeypatch.setitem(
        discord_bot.config.CONFIG_OVERRIDES,
        "DISCORD_COMMAND_RATE_LIMIT_SECONDS",
        1.0,
    )

    user = SimpleNamespace(id="user-123")
    discord_bot.reset_command_counts(user.id)

    assert await discord_bot.check_command_rate_limit(user)
    assert await discord_bot.check_command_rate_limit(user)
    assert not await discord_bot.check_command_rate_limit(user)

    clock.advance(1.1)

    assert await discord_bot.check_command_rate_limit(user)

    discord_bot.reset_command_counts(user.id)


@pytest.mark.unit
def test_default_human_message_broadcast_reads_config_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        discord_bot.config.CONFIG_OVERRIDES,
        "DISCORD_DEFAULT_BROADCAST",
        "true",
    )
    assert discord_bot._default_human_message_broadcast() is True

    monkeypatch.setitem(
        discord_bot.config.CONFIG_OVERRIDES,
        "DISCORD_DEFAULT_BROADCAST",
        "false",
    )
    assert discord_bot._default_human_message_broadcast() is False

    discord_bot.config.CONFIG_OVERRIDES.pop("DISCORD_DEFAULT_BROADCAST", None)
