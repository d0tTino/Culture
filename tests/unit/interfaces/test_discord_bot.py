import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class DummyCtx:
    def __init__(self) -> None:
        self.send = AsyncMock()


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
        Color=SimpleNamespace(blue=lambda: None),
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_say_command(discord_module: object) -> None:
    ctx = DummyCtx()
    await discord_module.say(ctx, message="hello")
    ctx.send.assert_awaited_once_with("Simulated message received: hello")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stats_command(discord_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discord_module, "get_llm_latency", lambda: 12.3)
    monkeypatch.setattr(discord_module, "get_kb_size", lambda: 7)
    ctx = DummyCtx()
    await discord_module.stats(ctx)
    ctx.send.assert_awaited_once_with("LLM latency: 12.3 ms; KB size: 7")


@pytest.mark.unit
def test_embed_creators(discord_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbed:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def set_author(self, *args: object, **kwargs: object) -> None:
            pass

        def add_field(self, *args: object, **kwargs: object) -> None:
            pass

    dummy_color = SimpleNamespace(
        blue=lambda: "blue",
        green=lambda: "green",
        gold=lambda: "gold",
        purple=lambda: "purple",
        teal=lambda: "teal",
        dark_green=lambda: "dark_green",
        dark_orange=lambda: "dark_orange",
        light_grey=lambda: "grey",
        red=lambda: "red",
        dark_teal=lambda: "dark_teal",
    )
    monkeypatch.setattr(
        discord_module, "discord", SimpleNamespace(Embed=DummyEmbed, Color=dummy_color)
    )
    bot = object.__new__(discord_module.SimulationDiscordBot)
    assert isinstance(
        discord_module.SimulationDiscordBot.create_step_start_embed(bot, 1), DummyEmbed
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_step_end_embed(bot, 2), DummyEmbed
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_knowledge_board_embed(bot, "a", "msg", 3),
        DummyEmbed,
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_role_change_embed(bot, "a", "old", "new", 4),
        DummyEmbed,
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_project_embed(
            bot, "create", "pname", "pid", "a", 5
        ),
        DummyEmbed,
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_agent_message_embed(bot, "a", "hello", step=6),
        DummyEmbed,
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_ip_change_embed(bot, "a", 1, 2, "reason", 7),
        DummyEmbed,
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_du_change_embed(
            bot, "a", 1.0, 2.0, "reason", 8
        ),
        DummyEmbed,
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_agent_action_embed(bot, "a", "idle", step=9),
        DummyEmbed,
    )
    assert isinstance(
        discord_module.SimulationDiscordBot.create_map_action_embed(
            bot, "a", "move", {"position": "x"}, 10
        ),
        DummyEmbed,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_forward_agent_messages_embed(discord_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbed:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.kwargs = kwargs

        def set_author(self, *args: object, **kwargs: object) -> None:
            pass

        def add_field(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(discord_module, "discord", SimpleNamespace(Embed=DummyEmbed))
    bot = object.__new__(discord_module.SimulationDiscordBot)
    bot.message_queue = asyncio.Queue()
    bot.user_channels = {}
    bot.user_agents = {}
    bot.is_ready = True

    async def fake_send_simulation_update(**kwargs: object) -> None:
        fake_send_simulation_update.kwargs = kwargs

    bot.send_simulation_update = fake_send_simulation_update  # type: ignore
    msg = discord_module.AgentMessage(
        agent_id="agent12345678",
        content="ignored",
        step=1,
        extra={"embed": {"title": "t", "description": "d", "color": 0x1}},
    )
    await bot.message_queue.put(msg)
    task = asyncio.create_task(discord_module.SimulationDiscordBot._forward_agent_messages(bot))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert fake_send_simulation_update.kwargs["embed"] is not None
    assert fake_send_simulation_update.kwargs["content"] is None
