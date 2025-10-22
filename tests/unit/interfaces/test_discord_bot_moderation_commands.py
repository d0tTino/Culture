import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


class RecordingCommandTree:
    def __init__(self, client: object) -> None:
        self.client = client
        self.commands: dict[str, object] = {}

    def command(self, *args: object, **kwargs: object):
        name = kwargs.get("name")

        def decorator(func):
            cmd_name = name or getattr(func, "__name__", "command")
            self.commands[cmd_name] = func
            return func

        return decorator

    def add_check(self, *args: object, **kwargs: object) -> None:
        return None

    async def sync(self) -> None:  # pragma: no cover - not exercised
        return None


class DummySimulationEvent:
    def __init__(self, *, type: str, data: dict[str, object]) -> None:
        self.type = type
        self.data = data


class DummyContext:
    def __init__(self) -> None:
        self.sim_state: dict[str, object] = {}
        self._event_queue: asyncio.Queue[DummySimulationEvent] = asyncio.Queue()
        self._event_queue_loop = None
        self.message_queue: asyncio.Queue[object] = asyncio.Queue()

    def get_event_queue(self) -> asyncio.Queue[DummySimulationEvent]:
        return self._event_queue


class DummyClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.http = object()
        self.user = SimpleNamespace()

    def get_channel(self, *_: object, **__: object) -> None:
        return None

    def event(self, func):
        return func


def reload_module(monkeypatch: pytest.MonkeyPatch):
    class DummyBot:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.tree = SimpleNamespace(command=lambda *a, **k: (lambda f: f))

        def command(self, *args: object, **kwargs: object):
            def decorator(func):
                return func

            return decorator

    dummy_discord = SimpleNamespace(
        Intents=SimpleNamespace(default=lambda: SimpleNamespace(message_content=True)),
        Client=DummyClient,
        Embed=lambda *a, **k: None,
        Color=SimpleNamespace(blue=lambda: 0),
        DiscordException=Exception,
        app_commands=SimpleNamespace(
            CommandTree=RecordingCommandTree,
            describe=lambda *a, **k: (lambda f: f),
        ),
    )

    dummy_app = SimpleNamespace(
        spawn_agent_command=AsyncMock(),
        start_simulation=AsyncMock(),
        stop_simulation=AsyncMock(),
    )

    dummy_ledger = SimpleNamespace(
        ledger=SimpleNamespace(get_balance_async=AsyncMock()),
        log_penalty=lambda *a, **k: None,
    )

    dummy_dashboard = SimpleNamespace(
        DEFAULT_CONTEXT=DummyContext(),
        SNAPSHOT_DIR="/tmp",
        AgentMessage=SimpleNamespace,
        SimulationEvent=DummySimulationEvent,
        message_sse_queue=object(),
        get_event_queue=lambda: asyncio.Queue(),
    )

    dummy_metrics = SimpleNamespace(get_llm_latency=lambda: 0, get_kb_size=lambda: 0)

    async def _evaluate_with_opa(payload: object):
        return True, payload

    dummy_policy = SimpleNamespace(
        allow_message=lambda *_: True,
        evaluate_with_opa=_evaluate_with_opa,
    )

    monkeypatch.setitem(sys.modules, "discord", dummy_discord)
    monkeypatch.setitem(sys.modules, "discord.app_commands", dummy_discord.app_commands)
    monkeypatch.setitem(sys.modules, "discord.ext", SimpleNamespace(commands=SimpleNamespace(Bot=DummyBot)))
    monkeypatch.setitem(sys.modules, "discord.ext.commands", SimpleNamespace(Bot=DummyBot))
    monkeypatch.setitem(sys.modules, "src.app", dummy_app)
    monkeypatch.setitem(
        sys.modules,
        "src.infra.config",
        SimpleNamespace(get_config=lambda *a, **k: None, SNAPSHOT_COMPRESS=False),
    )
    monkeypatch.setitem(sys.modules, "src.infra.ledger", dummy_ledger)
    monkeypatch.setitem(sys.modules, "src.interfaces.dashboard_backend", dummy_dashboard)
    monkeypatch.setitem(sys.modules, "src.interfaces.metrics", dummy_metrics)
    monkeypatch.setitem(sys.modules, "src.sim.context", SimpleNamespace(SimulationContext=DummyContext))
    monkeypatch.setitem(sys.modules, "src.utils.policy", dummy_policy)

    sys.modules.pop("src.interfaces.discord_moderation", None)
    sys.modules.pop("src.interfaces.discord_bot", None)
    module = importlib.import_module("src.interfaces.discord_bot")
    return module


@pytest.fixture()
def discord_module(monkeypatch: pytest.MonkeyPatch):
    module = reload_module(monkeypatch)
    module.DEFAULT_CONTEXT.sim_state = {}
    module.DEFAULT_CONTEXT._event_queue = asyncio.Queue()
    module.DEFAULT_CONTEXT.message_queue = asyncio.Queue()
    yield module
    importlib.reload(module)


class DummyInteraction:
    def __init__(self, *, admin: bool, user_id: str = "user") -> None:
        self.channel = SimpleNamespace(id=42)
        self.user = SimpleNamespace(
            id=user_id,
            guild_permissions=SimpleNamespace(administrator=admin),
        )
        self.response = SimpleNamespace(send_message=AsyncMock())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_command_tree_registers_moderation_commands(discord_module: object) -> None:
    context = discord_module.DEFAULT_CONTEXT
    bot = discord_module.SimulationDiscordBot("token", 123, context=context)
    tree = next(iter(bot.command_trees.values()))
    for name in {"reset_memory", "penalty", "mute", "unmute"}:
        assert name in tree.commands
        assert hasattr(tree.commands[name], "__wrapped__")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reset_memory_and_penalty_commands_require_admin(discord_module: object) -> None:
    context = discord_module.DEFAULT_CONTEXT
    bot = discord_module.SimulationDiscordBot("token", 123, context=context)
    tree = next(iter(bot.command_trees.values()))

    reset_cmd = tree.commands["reset_memory"]
    unauthorized = DummyInteraction(admin=False, user_id="user-reset-unauth")
    await reset_cmd(unauthorized, "agent-7")
    unauthorized.response.send_message.assert_awaited_once_with("unauthorized", ephemeral=True)
    assert bot.event_queue.empty()

    authorized = DummyInteraction(admin=True, user_id="user-reset-auth")
    await reset_cmd(authorized, "agent-7")
    event = await bot.event_queue.get()
    assert event.type == "moderation"
    assert event.data == {"command": "reset_memory", "agent_id": "agent-7"}
    authorized.response.send_message.assert_awaited_once_with(
        "memory reset", ephemeral=True
    )

    penalty_cmd = tree.commands["penalty"]
    unauthorized_penalty = DummyInteraction(admin=False, user_id="user-penalty-unauth")
    await penalty_cmd(unauthorized_penalty, "agent-5", 1.0, 2.0)
    unauthorized_penalty.response.send_message.assert_awaited_once_with(
        "unauthorized", ephemeral=True
    )
    assert bot.event_queue.empty()

    authorized_penalty = DummyInteraction(admin=True, user_id="user-penalty-auth")
    await penalty_cmd(authorized_penalty, "agent-5", 3.0, 4.5)
    penalty_event = await bot.event_queue.get()
    assert penalty_event.type == "moderation"
    assert penalty_event.data == {
        "command": "penalty",
        "agent_id": "agent-5",
        "ip": 3.0,
        "du": 4.5,
    }
    authorized_penalty.response.send_message.assert_awaited_once_with(
        "penalty applied", ephemeral=True
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mute_command_respects_rate_limiting(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = discord_module.DEFAULT_CONTEXT
    bot = discord_module.SimulationDiscordBot("token", 123, context=context)
    tree = next(iter(bot.command_trees.values()))
    mute_cmd = tree.commands["mute"]
    queue = context.get_event_queue()
    assert queue is bot.event_queue
    assert context.get_event_queue() is queue
    queue_put = AsyncMock()
    monkeypatch.setattr(queue, "put", queue_put)

    moderation_module = importlib.import_module("src.interfaces.discord_moderation")
    moderation_module._ACTION_COUNTS.clear()
    moderation_module._COOLDOWNS.clear()
    penalty_logger = Mock()
    monkeypatch.setattr(moderation_module, "log_penalty", penalty_logger)

    allowed = DummyInteraction(admin=False, user_id="user-mute-allowed")
    await mute_cmd(allowed, "agent-allowed")
    assert queue_put.await_count == 1
    allowed_event = queue_put.await_args_list[0].args[0]
    assert allowed_event.type == "moderation"
    assert allowed_event.data == {"command": "mute", "agent_id": "agent-allowed"}
    allowed.response.send_message.assert_awaited_once_with("muted", ephemeral=True)
    queue_put.reset_mock()

    assert discord_module.get_active_bot() is bot

    rate_limited = DummyInteraction(admin=False, user_id="user-mute-rate")
    monkeypatch.setattr(
        moderation_module,
        "evaluate_with_opa",
        AsyncMock(return_value=(False, None)),
    )
    await mute_cmd(rate_limited, "agent-rate")
    rate_limited.response.send_message.assert_awaited_once_with(
        "rate limited", ephemeral=True
    )
    assert moderation_module._ACTION_COUNTS["user-mute-rate:mute"] == 1
    assert queue_put.await_count == 0
    penalty_logger.assert_called_once()
