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
    return importlib.reload(importlib.import_module("src.interfaces.discord_bot"))


@pytest.fixture()
def discord_module(monkeypatch: pytest.MonkeyPatch):
    module = reload_module(monkeypatch)
    yield module
    importlib.reload(module)


class _Bus:
    def __init__(self) -> None:
        self.envelopes = []

    async def dispatch(self, envelope):
        self.envelopes.append(envelope)
        return SimpleNamespace(status="ok", user_message="ok")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_message_adapts_dm_to_direct_message(discord_module, monkeypatch):
    monkeypatch.setattr(discord_module, "allow_message", lambda _: True)
    monkeypatch.setattr(discord_module, "evaluate_with_opa", lambda content: asyncio.sleep(0, result=(True, content)))

    bus = _Bus()
    monkeypatch.setattr(discord_module, "get_command_bus", lambda _ctx: bus)

    from src.sim.context import SimulationContext

    bot = discord_module.SimulationDiscordBot("token", 123, context=SimulationContext())
    message = SimpleNamespace(content="/dm agent-z hello", author=SimpleNamespace(id="u1"), channel=SimpleNamespace(id=1))
    await bot.client.on_message(message)

    assert len(bus.envelopes) == 1
    env = bus.envelopes[0]
    assert env.intent == "direct_message"
    assert env.content == "hello"
    assert env.routing.recipient_id == "agent-z"


@pytest.mark.unit
def test_routing_parser_imported_from_transport_adapter(discord_module):
    from src.interfaces.transport_adapters import parse_discord_message_routing

    assert discord_module.parse_discord_message_routing is parse_discord_message_routing


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_message_replies_with_validation_error_for_empty_broadcast(discord_module, monkeypatch):
    monkeypatch.setattr(discord_module, "allow_message", lambda _: True)

    async def _eval(content: str):
        return True, content

    sent_messages: list[str] = []

    async def _send_channel_message(channel, *, content=None, embed=None):
        if content:
            sent_messages.append(content)

    monkeypatch.setattr(discord_module, "evaluate_with_opa", _eval)
    monkeypatch.setattr(discord_module, "send_channel_message", _send_channel_message)

    from src.sim.context import SimulationContext

    bot = discord_module.SimulationDiscordBot("token", 123, context=SimulationContext())
    message = SimpleNamespace(content="/broadcast   ", author=SimpleNamespace(id="human-1"), channel=SimpleNamespace(id=123))
    await bot.client.on_message(message)

    assert sent_messages == ["Broadcast message cannot be empty. Use /broadcast <message>."]
