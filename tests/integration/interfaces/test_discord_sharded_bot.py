from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("discord")

from src.interfaces import metrics
from src.interfaces.discord_sharded_bot import SimulationShardedDiscordBot, say, stats

sent_messages: list[str] = []


class DummyShardedClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.event = lambda f: f
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> object:  # pragma: no cover - minimal
        class DummyChannel:
            async def send(self_inner, *args: object, **kwargs: object) -> None:
                sent_messages.append(args[0] if args else "embed")

        return DummyChannel()

    async def start(self, token: str) -> None:
        self.token = token

    async def close(self) -> None:
        pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sharded_start_and_send() -> None:
    with patch(
        "src.interfaces.discord_sharded_bot.discord.AutoShardedClient",
        DummyShardedClient,
    ):
        bot = SimulationShardedDiscordBot("tok", 456)
        await bot.run_bot()
        await bot.send_simulation_update(content="hello")
        await bot.stop_bot()

    assert sent_messages
    sent_messages.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_commands() -> None:
    ctx = MagicMock()
    ctx.send = AsyncMock()
    await say.callback(ctx, message="hi")
    ctx.send.assert_awaited_once_with("Simulated message received: hi")

    ctx.send.reset_mock()
    metrics.LLM_LATENCY_MS.set(1.0)
    metrics.KNOWLEDGE_BOARD_SIZE.set(2)
    await stats.callback(ctx)
    ctx.send.assert_awaited_once_with("LLM latency: 1.0 ms; KB size: 2")
