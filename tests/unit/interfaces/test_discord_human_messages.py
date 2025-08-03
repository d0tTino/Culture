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
    dummy_discord = SimpleNamespace(
        Intents=SimpleNamespace(default=lambda: SimpleNamespace(message_content=True)),
        Client=DummyClient,
        Embed=object,
        TextChannel=object,
        Thread=object,
        DiscordException=Exception,
        Color=SimpleNamespace(blue=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "discord", dummy_discord)
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
