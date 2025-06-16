from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("discord")
from src.interfaces import metrics
from src.interfaces.discord_bot import SimulationDiscordBot, say, stats

sent_by_token: list[str] = []


class DummyDiscordClient:
    """Simple stand-in for discord.Client."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.event = lambda func: func
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> object:  # pragma: no cover - minimal
        token = getattr(self, "token", None)

        class DummyChannel:
            async def send(self_inner, *args: object, **kwargs: object) -> None:
                if token is not None:
                    sent_by_token.append(token)

        return DummyChannel()

    async def start(self, token: str) -> None:  # pragma: no cover - not used
        self.token = token

    async def close(self) -> None:  # pragma: no cover - not used
        pass


@pytest.fixture
def simulation_bot() -> SimulationDiscordBot:
    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        bot = SimulationDiscordBot("token", 123)
    return bot


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multi_token_start_and_send() -> None:
    start_tokens: list[str] = []

    class RecordingClient(DummyDiscordClient):
        async def start(self, token: str) -> None:
            start_tokens.append(token)

    tokens = ["tok1", "tok2"]

    def lookup(aid: str) -> str:
        return tokens[1] if aid == "agent_b" else tokens[0]

    with patch("src.interfaces.discord_bot.discord.Client", RecordingClient):
        bot = SimulationDiscordBot(tokens, 123, token_lookup=lookup)
        await bot.run_bot()
        await bot.send_simulation_update(content="hi", agent_id="agent_b")
        await bot.stop_bot()

    assert start_tokens == tokens
    assert sent_by_token == ["tok2"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_say_command(simulation_bot: SimulationDiscordBot) -> None:
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await say.callback(ctx, message="hello")
    ctx.send.assert_awaited_once_with("Simulated message received: hello")


@pytest.mark.unit
@pytest.mark.integration
@pytest.mark.asyncio
async def test_stats_command(simulation_bot: SimulationDiscordBot) -> None:
    ctx = MagicMock()
    ctx.send = AsyncMock()
    metrics.LLM_LATENCY_MS.set(42.0)
    metrics.KNOWLEDGE_BOARD_SIZE.set(7)
    await stats.callback(ctx)
    ctx.send.assert_awaited_once_with("LLM latency: 42.0 ms; KB size: 7")
