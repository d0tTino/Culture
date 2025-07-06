import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("discord")
from src.interfaces import metrics
from src.interfaces.discord_bot import SimulationDiscordBot, say, stats


class SimulationEvent:
    def __init__(self, event_type: str, data: dict[str, object] | None = None) -> None:
        self.event_type = event_type
        self.data = data


class AgentMessage:
    def __init__(self, agent_id: str, content: str, step: int) -> None:
        self.agent_id = agent_id
        self.content = content
        self.step = step


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
        patch("src.interfaces.dashboard_backend.EventSourceResponse", object),
    ):
        bot = SimulationDiscordBot("token", 123)
        assert "on_message" in bot.client._events
        await bot.run_bot()

        on_msg = bot.client._events["on_message"]
        msg = MagicMock()
        msg.content = "hello"
        msg.author = "user1"
        await on_msg(msg)
        stored = await q_events.get()
        assert stored.event_type == "broadcast"
        assert (stored.data or {})["content"] == "hello"

        await q_msgs.put(AgentMessage(agent_id="agent1", content="hi", step=0))
        await asyncio.sleep(0)
        assert bot.client.channel.sent

        await bot.stop_bot()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_message_updates_agent_state(monkeypatch: pytest.MonkeyPatch) -> None:
    q_events: asyncio.Queue[SimulationEvent] = asyncio.Queue()
    q_msgs: asyncio.Queue[AgentMessage] = asyncio.Queue()

    class Client(DummyDiscordClient):
        pass

    with (
        patch("src.interfaces.discord_bot.discord.Client", Client),
        patch(
            "src.interfaces.discord_bot.event_queue",
            q_events,
        ),
        patch(
            "src.interfaces.discord_bot.message_sse_queue",
            q_msgs,
        ),
    ):
        bot = SimulationDiscordBot("token", 456)
        await bot.run_bot()
        on_msg = bot.client._events["on_message"]
        msg = MagicMock()
        msg.content = "hello world"
        msg.author = "userA"
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
