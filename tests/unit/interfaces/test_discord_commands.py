import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class DummyBot:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.tree = SimpleNamespace(command=lambda *a, **k: (lambda f: f))

    def command(self, *args: object, **kwargs: object):
        def decorator(func):
            return func

        return decorator


def reload_module(monkeypatch: pytest.MonkeyPatch):
    dummy_commands = SimpleNamespace(Bot=DummyBot)
    dummy_discord = SimpleNamespace(
        Intents=SimpleNamespace(default=lambda: SimpleNamespace(message_content=True)),
        Client=object,
        Embed=object,
        TextChannel=object,
        Thread=object,
        DiscordException=Exception,
        Color=SimpleNamespace(green=lambda: None, red=lambda: None),
        app_commands=SimpleNamespace(CommandTree=object),
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


class DummyInteraction:
    def __init__(self) -> None:
        self.response = SimpleNamespace(send_message=AsyncMock())
        self.channel = None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_start_broadcasts_success_embed(discord_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_start_embed=MagicMock(return_value="embed"),
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(discord_module, "start_simulation", AsyncMock())
    interaction = DummyInteraction()
    await discord_module.slash_start(interaction)
    bot.create_start_embed.assert_called_once_with(True)
    bot.send_simulation_update.assert_awaited_once_with(embed="embed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_start_broadcasts_failure_embed(discord_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_start_embed=MagicMock(return_value="embed"),
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(discord_module, "start_simulation", AsyncMock(side_effect=RuntimeError("boom")))
    interaction = DummyInteraction()
    await discord_module.slash_start(interaction)
    bot.create_start_embed.assert_called_once()
    args = bot.create_start_embed.call_args[0]
    assert args[0] is False and "boom" in args[1]
    bot.send_simulation_update.assert_awaited_once_with(embed="embed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_stop_broadcasts_success_embed(discord_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_stop_embed=MagicMock(return_value="embed"),
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(discord_module, "stop_simulation", AsyncMock())
    interaction = DummyInteraction()
    await discord_module.slash_stop(interaction)
    bot.create_stop_embed.assert_called_once_with(True)
    bot.send_simulation_update.assert_awaited_once_with(embed="embed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_spawn_broadcasts_success_embed(discord_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_spawn_embed=MagicMock(return_value="embed"),
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(discord_module, "spawn_agent_command", AsyncMock())
    interaction = DummyInteraction()
    await discord_module.slash_spawn(interaction, "agent")
    bot.create_spawn_embed.assert_called_once_with("agent", True)
    bot.send_simulation_update.assert_awaited_once_with(embed="embed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_spawn_broadcasts_failure_embed(discord_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_spawn_embed=MagicMock(return_value="embed"),
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(discord_module, "spawn_agent_command", AsyncMock(side_effect=RuntimeError("bad")))
    interaction = DummyInteraction()
    await discord_module.slash_spawn(interaction, "agent")
    bot.create_spawn_embed.assert_called_once()
    args = bot.create_spawn_embed.call_args[0]
    assert args[0] == "agent" and args[1] is False
    bot.send_simulation_update.assert_awaited_once_with(embed="embed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kb_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.sim.simulation import Simulation

    class DummyAgent:
        def get_id(self) -> str:
            return "a"

        agent_id = "a"
        state = SimpleNamespace(ip=10.0, du=10.0)

    sim = Simulation([DummyAgent()])
    sim.knowledge_board.add_entry = MagicMock()
    await sim._handle_human_command("/kb first")
    await sim._handle_human_command("/kb second")
    sim.knowledge_board.add_entry.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_message_relay_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.infra import ledger as ledger_module
    from src.sim.simulation import Simulation

    class DummyAgent:
        def get_id(self) -> str:
            return "a"

        agent_id = "a"
        state = SimpleNamespace(ip=10.0, du=10.0)

    sim = Simulation([])
    sim.agents = [DummyAgent()]
    monkeypatch.setattr(ledger_module.ledger, "spend", AsyncMock())
    await sim._handle_human_command("hello")
    await sim._handle_human_command("hello again")
    ledger_module.ledger.spend.assert_awaited_once()
