import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest


class DummyClient:
    def __init__(self, *args, **kwargs) -> None:
        self.user = SimpleNamespace(id="bot")

    def event(self, func):
        setattr(self, func.__name__, func)
        return func


class DummyBot:
    def __init__(self, *args, **kwargs) -> None:
        self.tree = SimpleNamespace(command=lambda *a, **k: (lambda f: f))

    def command(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


def reload_module(monkeypatch: pytest.MonkeyPatch):
    dummy_commands = SimpleNamespace(Bot=DummyBot)
    class DummyTree:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def command(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    dummy_app_commands = SimpleNamespace(
        CommandTree=DummyTree,
        describe=lambda **_kwargs: (lambda func: func),
    )
    dummy_discord = SimpleNamespace(
        Intents=SimpleNamespace(default=lambda: SimpleNamespace(message_content=True)),
        Client=DummyClient,
        Embed=object,
        TextChannel=object,
        Thread=object,
        DiscordException=Exception,
        Color=SimpleNamespace(blue=lambda: None),
        app_commands=dummy_app_commands,
    )
    monkeypatch.setitem(sys.modules, "discord", dummy_discord)
    monkeypatch.setitem(sys.modules, "discord.app_commands", dummy_discord.app_commands)
    monkeypatch.setitem(sys.modules, "discord.ext", SimpleNamespace(commands=dummy_commands))
    monkeypatch.setitem(sys.modules, "discord.ext.commands", dummy_commands)
    module = importlib.reload(importlib.import_module("src.interfaces.discord_bot"))
    return module


@pytest.fixture()
def discord_module(monkeypatch: pytest.MonkeyPatch):
    module = reload_module(monkeypatch)
    yield module
    importlib.reload(module)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_human_messages_counter_increments(discord_module, monkeypatch):
    monkeypatch.setattr(discord_module, "allow_message", lambda _: True)

    async def dummy_eval(content):
        return True, content

    monkeypatch.setattr(discord_module, "evaluate_with_opa", dummy_eval)

    from src.interfaces import metrics
    from src.sim.context import SimulationContext

    bot = discord_module.SimulationDiscordBot("token", 123, context=SimulationContext())

    message = SimpleNamespace(
        content="hello",
        author=SimpleNamespace(id="user"),
        channel=SimpleNamespace(id=999),
    )

    before = metrics.HUMAN_MESSAGES_TOTAL._value.get()
    await bot.client.on_message(message)
    after = metrics.HUMAN_MESSAGES_TOTAL._value.get()

    assert after == before + 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_message_routes_plain_text_to_single_default_agent(discord_module, monkeypatch):
    monkeypatch.setattr(discord_module, "allow_message", lambda _: True)
    monkeypatch.setattr(
        discord_module,
        "evaluate_with_opa",
        lambda content: asyncio.sleep(0, result=(True, content)),
    )
    monkeypatch.setattr(
        discord_module.ledger,
        "get_balance_async",
        lambda _: asyncio.sleep(0, result=(10.0, 10.0)),
    )

    from src.sim.context import SimulationContext

    bot = discord_module.SimulationDiscordBot(
        "token", 123, context=SimulationContext(), channel_map={"agent-b": 456, "agent-a": 123}
    )
    bot.event_queue = asyncio.Queue()

    message = SimpleNamespace(
        content="hello everyone",
        author=SimpleNamespace(id="new-user"),
        channel=SimpleNamespace(id=888, send=lambda *_: asyncio.sleep(0)),
    )
    await bot.client.on_message(message)
    evt = await bot.event_queue.get()

    assert evt.type == "broadcast"
    assert evt.data["target_agent_id"] == "agent-a"
    assert evt.data["broadcast"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_message_parses_explicit_dm_target(discord_module, monkeypatch):
    monkeypatch.setattr(discord_module, "allow_message", lambda _: True)

    async def _eval(content: str):
        return True, content

    async def _bal(agent_id: str):
        return (10.0, 10.0)

    monkeypatch.setattr(discord_module, "evaluate_with_opa", _eval)
    monkeypatch.setattr(discord_module.ledger, "get_balance_async", _bal)

    from src.sim.context import SimulationContext

    bot = discord_module.SimulationDiscordBot("token", 123, context=SimulationContext())
    bot.event_queue = asyncio.Queue()

    message = SimpleNamespace(
        content="/dm agent-z hi there",
        author=SimpleNamespace(id="human-1"),
        channel=SimpleNamespace(id=123, send=lambda *_: asyncio.sleep(0)),
    )
    await bot.client.on_message(message)
    evt = await bot.event_queue.get()

    assert evt.data["recipient_id"] == "agent-z"
    assert evt.data["target_agent_id"] == "agent-z"
    assert evt.data["broadcast"] is False
    assert evt.data["content"] == "hi there"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected_broadcast"),
    [
        ("@agent-z: hi mention", False),
        ("/dm agent-z hi slash", False),
        ("/broadcast hi all", True),
    ],
)
async def test_on_message_precheck_costs_align_with_routing(
    discord_module,
    monkeypatch,
    content,
    expected_broadcast,
):
    monkeypatch.setattr(discord_module, "allow_message", lambda _: True)

    async def _eval(raw: str):
        return True, raw

    def _config_lookup(key: str):
        values = {
            "IP_COST_BROADCAST_MESSAGE": 6.0,
            "IP_COST_SEND_DIRECT_MESSAGE": 2.0,
            "DU_COST_BROADCAST_ACTION": 7.0,
            "DU_COST_PER_ACTION": 3.0,
        }
        return values.get(key)

    balance_checks: list[str] = []

    async def _bal(agent_id: str):
        balance_checks.append(agent_id)
        return (4.0, 4.0)

    sent_messages: list[str] = []

    async def _send_channel_message(channel, *, content=None, embed=None):
        if content:
            sent_messages.append(content)

    monkeypatch.setattr(discord_module, "evaluate_with_opa", _eval)
    monkeypatch.setattr(discord_module.config, "get_config", _config_lookup)
    monkeypatch.setattr(discord_module.ledger, "get_balance_async", _bal)
    monkeypatch.setattr(discord_module, "send_channel_message", _send_channel_message)

    from src.sim.context import SimulationContext

    bot = discord_module.SimulationDiscordBot(
        "token", 123, context=SimulationContext(), channel_map={"agent-z": 999}
    )
    bot.event_queue = asyncio.Queue()

    message = SimpleNamespace(
        content=content,
        author=SimpleNamespace(id="human-1"),
        channel=SimpleNamespace(id=123),
    )
    await bot.client.on_message(message)

    assert balance_checks == ["agent-z"]
    if expected_broadcast:
        assert sent_messages == ["Insufficient IP/DU"]
        assert bot.event_queue.empty()
    else:
        assert sent_messages == []
        evt = await bot.event_queue.get()
        assert evt.data["broadcast"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_message_parses_explicit_broadcast_prefix(discord_module, monkeypatch):
    monkeypatch.setattr(discord_module, "allow_message", lambda _: True)

    async def _eval(content: str):
        return True, content

    async def _bal(agent_id: str):
        return (10.0, 10.0)

    monkeypatch.setattr(discord_module, "evaluate_with_opa", _eval)
    monkeypatch.setattr(discord_module.ledger, "get_balance_async", _bal)

    from src.sim.context import SimulationContext

    bot = discord_module.SimulationDiscordBot(
        "token", 123, context=SimulationContext(), channel_map={"agent-z": 999}
    )
    bot.event_queue = asyncio.Queue()

    message = SimpleNamespace(
        content="/broadcast hello all",
        author=SimpleNamespace(id="human-1"),
        channel=SimpleNamespace(id=123, send=lambda *_: asyncio.sleep(0)),
    )
    await bot.client.on_message(message)
    evt = await bot.event_queue.get()

    assert evt.data["target_agent_id"] is not None
    assert evt.data["broadcast"] is True
    assert evt.data["content"] == "hello all"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_message_parses_explicit_mention_target(discord_module, monkeypatch):
    monkeypatch.setattr(discord_module, "allow_message", lambda _: True)

    async def _eval(content: str):
        return True, content

    async def _bal(agent_id: str):
        return (10.0, 10.0)

    monkeypatch.setattr(discord_module, "evaluate_with_opa", _eval)
    monkeypatch.setattr(discord_module.ledger, "get_balance_async", _bal)

    from src.sim.context import SimulationContext

    bot = discord_module.SimulationDiscordBot("token", 123, context=SimulationContext())
    bot.event_queue = asyncio.Queue()

    message = SimpleNamespace(
        content="@agent-z: hi there",
        author=SimpleNamespace(id="human-1"),
        channel=SimpleNamespace(id=123, send=lambda *_: asyncio.sleep(0)),
    )
    await bot.client.on_message(message)
    evt = await bot.event_queue.get()

    assert evt.data["recipient_id"] == "agent-z"
    assert evt.data["target_agent_id"] == "agent-z"
    assert evt.data["broadcast"] is False
    assert evt.data["content"] == "hi there"
