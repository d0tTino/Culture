from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("discord")

from src.interfaces.discord_bot import SimulationDiscordBot, say, stats


class DummyDiscordClient:
    """Simple stand-in for discord.Client."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.event = lambda func: func
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> None:
        return None

    async def start(self, token: str) -> None:  # pragma: no cover - not used
        pass

    async def close(self) -> None:  # pragma: no cover - not used
        pass


@pytest.fixture
def simulation_bot() -> SimulationDiscordBot:
    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        bot = SimulationDiscordBot("token", 123)
    return bot


@pytest.mark.integration
@pytest.mark.asyncio
async def test_say_command(simulation_bot: SimulationDiscordBot) -> None:
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await say.callback(ctx, message="hello")
    ctx.send.assert_awaited_once_with("Simulated message received: hello")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stats_command(simulation_bot: SimulationDiscordBot) -> None:
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await stats.callback(ctx)
    ctx.send.assert_awaited_once_with("LLM latency: 0 ms; KB size: 0")
