import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.interfaces import dashboard_backend as db


@pytest.fixture
def anyio_backend() -> str:
    """Limit AnyIO to the asyncio backend."""
    return "asyncio"


class DummyDiscordClient:
    """Minimal stand-in for ``discord.Client``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.event = lambda f: f
        self.user = "dummy"

    async def start(self, token: str) -> None:  # pragma: no cover - unused
        self.token = token

    async def close(self) -> None:  # pragma: no cover - unused
        pass


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
@pytest.mark.anyio("asyncio")
async def test_event_queue_broadcast_cost(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    agent = DummyAgent("A")
    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        sim = Simulation([agent])

    queue = db.get_event_queue()
    await queue.put(
        db.SimulationEvent(
            type="broadcast", data={"author": "human", "content": "/broadcast hello"}
        )
    )
    await asyncio.sleep(0.1)

    assert agent.state.ip == pytest.approx(1.0)
    assert agent.state.du == pytest.approx(1.0)
    async with sim._msg_lock:
        recipients = {m["recipient_id"] for m in sim.pending_messages_for_next_round}
    assert recipients == {"A"}

    sim.close()
    await queue.put(None)


@pytest.mark.integration
@pytest.mark.anyio("asyncio")
async def test_event_queue_kb_post(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sys.modules.setdefault("neo4j", DummyNeo4j())
    from src.infra.ledger import Ledger
    from src.sim.simulation import Simulation

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr("src.infra.ledger.ledger", ledger)
    monkeypatch.setattr("src.sim.simulation.ledger", ledger)

    agent = DummyAgent("A")
    with patch("src.interfaces.discord_bot.discord.Client", DummyDiscordClient):
        sim = Simulation([agent])

    queue = db.get_event_queue()
    await queue.put(
        db.SimulationEvent(
            type="broadcast", data={"author": "human", "content": "/kb important fact"}
        )
    )
    await asyncio.sleep(0.1)

    entries = sim.knowledge_board.get_state(max_entries=1)
    assert "important fact" in entries[0]

    sim.close()
    await queue.put(None)


@pytest.mark.integration
@pytest.mark.anyio("asyncio")
async def test_user_channel_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.interfaces.discord_bot import SimulationDiscordBot
    from src.sim.context import SimulationContext

    class RecordingChannel:
        def __init__(self, cid: int) -> None:
            self.id = cid
            self.sent: list[object] = []

        async def send(self, *args: object, **kwargs: object) -> None:
            if args:
                self.sent.append(args[0])
            elif "embed" in kwargs:
                self.sent.append(kwargs["embed"])
            else:
                self.sent.append(None)

    class RecordingClient(DummyDiscordClient):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self._events: dict[str, object] = {}
            self.channels: dict[int, RecordingChannel] = {}

            def _event(func: object) -> object:
                if hasattr(func, "__name__"):
                    self._events[func.__name__] = func
                return func

            self.event = _event

        def get_channel(self, channel_id: int) -> RecordingChannel:
            channel = self.channels.setdefault(channel_id, RecordingChannel(channel_id))
            return channel

    q_events: asyncio.Queue[db.SimulationEvent] = asyncio.Queue()
    q_msgs: asyncio.Queue[db.AgentMessage] = asyncio.Queue()
    ctx = SimulationContext()
    ctx._event_queue = q_events
    ctx._event_queue_loop = asyncio.get_event_loop()
    ctx.message_queue = q_msgs

    with (
        patch("src.interfaces.discord_bot.discord.Client", RecordingClient),
        patch(
            "src.interfaces.discord_bot.evaluate_with_opa",
            AsyncMock(side_effect=lambda c: (True, c)),
        ),
    ):
        bot = await SimulationDiscordBot.create("token", 999, context=ctx)
        tasks = bot.run_bot()
        await asyncio.gather(*tasks[:-1])

        on_msg = bot.client._events["on_message"]
        msg = MagicMock()
        msg.content = "hi"
        msg.author = MagicMock()
        msg.author.id = 42
        msg.channel = MagicMock()
        msg.channel.id = 111
        await on_msg(msg)

        await q_msgs.put(
            db.AgentMessage(agent_id="agent1", content="hello", step=0, recipient_id="42")
        )
        await asyncio.sleep(0)

        assert 111 in bot.client.channels
        channel = bot.client.channels[111]
        assert channel.sent and channel.sent[0] == "hello"

        assert 999 not in bot.client.channels

        await bot.stop_bot()
