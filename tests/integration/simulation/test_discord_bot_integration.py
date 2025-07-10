import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.interfaces import dashboard_backend as db
from src.interfaces.discord_bot import SimulationDiscordBot
from src.sim.simulation import Simulation


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
            def __init__(self) -> None:
                self.sent: list[tuple[tuple[object, ...], dict[str, object]]] = []

            async def send(self_inner, *args: object, **kwargs: object) -> None:
                self_inner.sent.append((args, kwargs))

        self.channel = DummyChannel()
        return self.channel

    async def start(self, token: str) -> None:  # pragma: no cover - not used
        self.token = token

    async def close(self) -> None:  # pragma: no cover - not used
        pass


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = SimpleNamespace(
            ip=2.0,
            du=2.0,
            short_term_memory=[],
            messages_sent_count=0,
            last_message_step=None,
            collective_ip=0.0,
            collective_du=0.0,
        )
        self.received: list[dict] | None = None

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, state: SimpleNamespace) -> None:
        self.state = state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict | None = None,
        vector_store_manager=None,
        knowledge_board=None,
    ) -> dict:
        self.received = (
            environment_perception.get("perceived_messages", []) if environment_perception else []
        )
        msg = db.AgentMessage(agent_id=self.agent_id, content="ack", step=simulation_step)
        await db.enqueue_message(msg)
        return {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_simulation_bot_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    q_events: asyncio.Queue[db.SimulationEvent | None] = asyncio.Queue()
    q_msgs: asyncio.Queue[db.AgentMessage] = asyncio.Queue()
    handled: list[str] = []

    original_handle = Simulation._handle_human_command

    async def wrapped(self: Simulation, text: str) -> None:
        handled.append(text)
        await original_handle(self, text)

    class Client(DummyDiscordClient):
        pass

    with (
        patch("src.interfaces.discord_bot.discord.Client", Client),
        patch("src.interfaces.dashboard_backend.get_event_queue", lambda: q_events),
        patch("src.interfaces.dashboard_backend.message_sse_queue", q_msgs),
        patch("src.interfaces.discord_bot.message_sse_queue", q_msgs),
        patch.object(Simulation, "_handle_human_command", wrapped),
        patch("src.interfaces.dashboard_backend.EventSourceResponse", object),
    ):
        bot = SimulationDiscordBot("token", 1)
        agent = DummyAgent("A")
        sim = Simulation([agent], discord_bot=bot)

        await bot.run_bot()

        on_msg = bot.client._events["on_message"]
        msg = MagicMock()
        msg.content = "hello"
        msg.author = "user1"
        await on_msg(msg)
        await asyncio.sleep(0.05)

        await sim.run_step()
        await asyncio.sleep(0.05)

        assert handled == ["hello"]
        assert agent.received and agent.received[0]["content"] == "hello"
        assert bot.client.channel.sent and bot.client.channel.sent[0][0][0] == "ack"

        await bot.stop_bot()
