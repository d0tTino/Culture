import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("discord")

from src.interfaces.discord_bot import SimulationDiscordBot
from src.sim.context import SimulationContext


class DummyDiscordClient:
    """Minimal stand-in for ``discord.Client``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._events: dict[str, object] = {}

        def _event(func: object) -> object:
            if hasattr(func, "__name__"):
                self._events[func.__name__] = func
            return func

        self.event = _event
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> object:  # pragma: no cover - minimal
        class DummyChannel:
            async def send(self_inner, *args: object, **kwargs: object) -> None:
                pass

        self.channel = DummyChannel()
        return self.channel

    async def start(self, token: str) -> None:
        self.token = token

    async def close(self) -> None:  # pragma: no cover - not used
        pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_bots_message_routing() -> None:
    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()

    ctx1 = SimulationContext()
    ctx1._event_queue = q1
    ctx1._event_queue_loop = asyncio.get_event_loop()
    ctx1.message_queue = asyncio.Queue()

    ctx2 = SimulationContext()
    ctx2._event_queue = q2
    ctx2._event_queue_loop = asyncio.get_event_loop()
    ctx2.message_queue = asyncio.Queue()

    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        bot1 = await SimulationDiscordBot.create("token1", 101, context=ctx1)
        bot2 = await SimulationDiscordBot.create("token2", 202, context=ctx2)
        tasks = [*bot1.run_bot()[:-1], *bot2.run_bot()[:-1]]
        await asyncio.gather(*tasks)

        on_msg1 = bot1.client._events["on_message"]
        on_msg2 = bot2.client._events["on_message"]

        msg1 = MagicMock()
        msg1.content = "hello-1"
        msg1.author = "user1"
        await on_msg1(msg1)
        await asyncio.sleep(0)
        event1 = await q1.get()
        assert (event1.data or {}).get("content") == "hello-1"
        assert q2.empty()

        msg2 = MagicMock()
        msg2.content = "hello-2"
        msg2.author = "user2"
        await on_msg2(msg2)
        await asyncio.sleep(0)
        event2 = await q2.get()
        assert (event2.data or {}).get("content") == "hello-2"
        assert q1.empty()

        await bot1.stop_bot()
        await bot2.stop_bot()
