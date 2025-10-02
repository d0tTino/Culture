import asyncio
import sys
import typing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("discord")
from src.interfaces import metrics
from src.interfaces.dashboard_backend import AgentMessage, SimulationEvent
from src.interfaces.discord_bot import (
    SimulationDiscordBot,
    say,
    slash_spawn,
    slash_start,
    slash_stop,
    stats,
)
from src.sim.context import SimulationContext

sent_by_token: list[str] = []


class DummyDiscordClient:
    """Simple stand-in for discord.Client."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._events: dict[str, object] = {}

        def _event(func: object) -> object:
            if hasattr(func, "__name__"):
                self._events[func.__name__] = func
            return func

        self.event = _event
        self.user = "dummy"

    def get_channel(self, channel_id: int) -> object:  # pragma: no cover - minimal
        token = getattr(self, "token", None)

        class DummyChannel:
            def __init__(self) -> None:
                self.sent: list[tuple[tuple[object, ...], dict[str, object]]] = []

            async def send(self_inner, *args: object, **kwargs: object) -> None:
                self_inner.sent.append((args, kwargs))
                if token is not None:
                    sent_by_token.append(token)

        self.channel = DummyChannel()
        return self.channel

    async def start(self, token: str) -> None:  # pragma: no cover - not used
        self.token = token

    async def close(self) -> None:  # pragma: no cover - not used
        pass


@pytest.fixture
async def simulation_bot() -> SimulationDiscordBot:
    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        bot = await SimulationDiscordBot.create("token", 123, context=SimulationContext())
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

    with (
        patch("src.interfaces.discord_bot.discord.Client", RecordingClient),
        patch(
            "src.interfaces.discord_bot.evaluate_with_opa",
            AsyncMock(side_effect=lambda content: (True, content)),
        ),
    ):
        bot = await SimulationDiscordBot.create(
            tokens, 123, token_lookup=lookup, context=SimulationContext()
        )
        tasks = bot.run_bot()
        await asyncio.gather(*tasks[:-1])
        await bot.send_simulation_update(content="hi", agent_id="agent_b")
        await bot.stop_bot()

    assert start_tokens == tokens
    assert sent_by_token == ["tok2"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multi_token_message_forwarding() -> None:
    q_events: asyncio.Queue[SimulationEvent] = asyncio.Queue()
    q_msgs: asyncio.Queue[AgentMessage] = asyncio.Queue()

    class Client(DummyDiscordClient):
        pass

    tokens = ["tok1", "tok2"]

    ctx = SimulationContext()
    ctx._event_queue = q_events
    ctx._event_queue_loop = asyncio.get_event_loop()
    ctx.message_queue = q_msgs
    with (
        patch("src.interfaces.discord_bot.discord.Client", Client),
        patch("src.interfaces.dashboard_backend.get_event_queue", lambda: q_events),
        patch(
            "src.interfaces.discord_bot.ledger.get_balance_async",
            AsyncMock(return_value=(1.0, 1.0)),
        ),
    ):
        bot = await SimulationDiscordBot.create(tokens, 999, channel_map={"A": 999}, context=ctx)
        tasks = bot.run_bot()
        await asyncio.gather(*tasks[:-1])

        for token in tokens:
            assert "on_message" in bot.clients[token]._events
            on_msg = bot.clients[token]._events["on_message"]
            msg = MagicMock()
            msg.content = f"hello-{token}"
            msg.author = SimpleNamespace(id=f"user_{token}")
            msg.channel = SimpleNamespace(id=999, send=AsyncMock())
            await on_msg(msg)
            stored = await q_events.get()
            assert (stored.data or {}).get("content") == f"hello-{token}"

        await bot.stop_bot()


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_on_message_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    q_events: asyncio.Queue[SimulationEvent] = asyncio.Queue()
    q_msgs: asyncio.Queue[AgentMessage] = asyncio.Queue()

    class Client(DummyDiscordClient):
        pass

    with (
        patch("src.interfaces.discord_bot.discord.Client", Client),
        patch(
            "src.interfaces.dashboard_backend.get_event_queue",
            lambda: q_events,
        ),
        patch("src.interfaces.discord_bot.message_sse_queue", q_msgs),
        patch(
            "src.interfaces.discord_bot.ledger.get_balance_async",
            AsyncMock(return_value=(1.0, 1.0)),
        ),
        patch("src.interfaces.dashboard_backend.EventSourceResponse", object),
    ):
        bot = await SimulationDiscordBot.create(
            "token", 123, channel_map={"A": 123}, context=SimulationContext()
        )
        assert "on_message" in bot.client._events
        tasks = bot.run_bot()
        await asyncio.gather(*tasks[:-1])

        on_msg = bot.client._events["on_message"]
        msg = MagicMock()
        msg.content = "hello"
        msg.author = SimpleNamespace(id="user1")
        msg.channel = SimpleNamespace(id=123, send=AsyncMock())
        await on_msg(msg)
        stored = await q_events.get()
        assert stored.type == "broadcast"
        assert (stored.data or {})["content"] == "hello"

        await q_msgs.put(AgentMessage(agent_id="agent1", content="hi", step=0))
        await asyncio.sleep(0)
        assert bot.client.channel.sent

        await bot.stop_bot()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_control_commands_require_admin_without_opa(
    simulation_bot: SimulationDiscordBot,
) -> None:
    class Perms:
        administrator = False

    user = SimpleNamespace(id="user0", guild_permissions=Perms())

    async def _assert_unauthorized(
        command: typing.Callable[..., typing.Awaitable[None]], *args: object
    ) -> None:
        response = SimpleNamespace(send_message=AsyncMock())
        interaction = SimpleNamespace(
            user=user,
            response=response,
            channel=SimpleNamespace(id=simulation_bot.channel_id),
        )
        await command(interaction, *args)
        response.send_message.assert_awaited_once_with("unauthorized", ephemeral=True)

    await _assert_unauthorized(slash_start)
    await _assert_unauthorized(slash_stop)
    await _assert_unauthorized(slash_spawn, "agent-1")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_message_updates_agent_state(monkeypatch: pytest.MonkeyPatch) -> None:
    q_events: asyncio.Queue[SimulationEvent] = asyncio.Queue()
    q_msgs: asyncio.Queue[AgentMessage] = asyncio.Queue()

    class Client(DummyDiscordClient):
        pass

    ctx = SimulationContext()
    ctx._event_queue = q_events
    ctx._event_queue_loop = asyncio.get_event_loop()
    ctx.message_queue = q_msgs
    with (
        patch("src.interfaces.discord_bot.discord.Client", Client),
        patch(
            "src.interfaces.discord_bot.ledger.get_balance_async",
            AsyncMock(return_value=(1.0, 1.0)),
        ),
    ):
        bot = await SimulationDiscordBot.create("token", 456, channel_map={"A": 456}, context=ctx)
        tasks = bot.run_bot()
        await asyncio.gather(*tasks[:-1])
        on_msg = bot.client._events["on_message"]
        msg = MagicMock()
        msg.content = "hello world"
        msg.author = SimpleNamespace(id="userA")
        msg.channel = SimpleNamespace(id=456, send=AsyncMock())
        await on_msg(msg)
        event = await q_events.get()
        agent_state = {"messages": []}
        agent_state["messages"].append(event.data["content"])
        assert agent_state["messages"] == ["hello world"]
        await bot.stop_bot()


class DummyNeo4j:
    Driver = object
    GraphDatabase = object


class DummyState:
    def __init__(self, ip: float = 2.0, du: float = 2.0) -> None:
        self.ip = ip
        self.du = du
        self.short_term_memory = []
        self.messages_sent_count = 0
        self.last_message_step = None
        self.collective_ip = 0.0
        self.collective_du = 0.0


class DummyAgent:
    def __init__(self, agent_id: str, ip: float = 2.0, du: float = 2.0) -> None:
        self.agent_id = agent_id
        self._state = DummyState(ip, du)
        from src.infra.ledger import ledger as _ledger

        _ledger.log_change(agent_id, ip, du, "init")

    def get_id(self) -> str:
        return self.agent_id

    @property
    def state(self) -> DummyState:
        return self._state

    def update_state(self, state: DummyState) -> None:
        self._state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        return {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_broadcast_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    agents = [DummyAgent("A"), DummyAgent("B"), DummyAgent("C")]
    sim = Simulation(agents)

    await sim._handle_human_command("/broadcast hello all")

    async with sim._msg_lock:
        recipients = {m["recipient_id"] for m in sim.pending_messages_for_next_round}

    assert recipients == {"A", "B", "C"}
    assert agents[0].state.ip == pytest.approx(1.0)
    assert agents[0].state.du == pytest.approx(1.0)
