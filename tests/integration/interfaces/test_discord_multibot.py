import asyncio
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("discord")

from src.infra import config
from src.interfaces import token_sql, token_store
from src.interfaces.dashboard_backend import AgentMessage, SimulationEvent
from src.interfaces.discord_bot import SimulationDiscordBot
from src.sim.context import SimulationContext


class DummyDiscordClient:
    """Simple stand-in for ``discord.Client``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._events: dict[str, object] = {}

        def _event(func: object) -> object:
            if hasattr(func, "__name__"):
                self._events[func.__name__] = func
            return func

        self.event = _event
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> object:
        class DummyChannel:
            def __init__(self) -> None:
                self.id = channel_id
                self.sent: list[tuple[tuple[object, ...], dict[str, object]]] = []

            async def send(self_inner, *args: object, **kwargs: object) -> None:
                self_inner.sent.append((args, kwargs))

        self.channel = DummyChannel()
        return self.channel

    async def start(self, token: str) -> None:
        self.token = token

    async def close(self) -> None:
        pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multibot_channels_from_db(monkeypatch: pytest.MonkeyPatch) -> None:
    q_events: asyncio.Queue[SimulationEvent] = asyncio.Queue()
    q_msgs: asyncio.Queue[AgentMessage] = asyncio.Queue()

    db_url = "sqlite+aiosqlite:///:memory:"
    monkeypatch.setenv("DISCORD_TOKENS_DB_URL", db_url)
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "DISCORD_TOKENS_DB_URL", db_url)
    config.load_config(validate_required=False)

    token_sql._engine = None  # type: ignore[attr-defined]
    token_sql._sessionmaker = None  # type: ignore[attr-defined]
    await token_sql.save_token("agent_a", "tok_a")
    await token_sql.save_token("agent_b", "tok_b")

    monkeypatch.setattr(token_store, "list_tokens", token_sql.list_tokens)
    monkeypatch.setattr(token_store, "lookup_token", token_sql.get_token)

    ctx = SimulationContext()
    ctx._event_queue = q_events
    ctx._event_queue_loop = asyncio.get_event_loop()
    ctx.message_queue = q_msgs

    channel_map = {"agent_a": 101, "agent_b": 202}

    with (
        patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient),
        patch("src.interfaces.dashboard_backend.get_event_queue", lambda: q_events),
    ):
        bot = await SimulationDiscordBot.create(None, 999, channel_map=channel_map, context=ctx)
        tasks = bot.run_bot()
        await asyncio.gather(*tasks[:-1])

        assert set(bot.clients.keys()) == {"tok_a", "tok_b"}

        for agent, token in [("agent_a", "tok_a"), ("agent_b", "tok_b")]:
            on_msg = bot.clients[token]._events["on_message"]
            msg = MagicMock()
            msg.content = f"hello-{token}"
            msg.author = f"user_{token}"
            msg.channel = MagicMock()
            msg.channel.id = channel_map[agent]
            await on_msg(msg)
            stored = await q_events.get()
            assert stored.data.get("recipient_id") == agent
            assert stored.data.get("content") == f"hello-{token}"

        await bot.stop_bot()
