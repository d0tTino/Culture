import asyncio
import importlib
import sys
from collections.abc import Callable
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

    class DummyCommandTree:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def command(
            self, *a: object, **k: object
        ) -> Callable[[Callable[..., object]], Callable[..., object]]:
            return lambda f: f

        def add_check(self, *a: object, **k: object) -> None:
            return None

    dummy_discord = SimpleNamespace(
        Intents=SimpleNamespace(default=lambda: SimpleNamespace(message_content=True)),
        Client=object,
        Embed=lambda *a, **k: None,
        TextChannel=object,
        Thread=object,
        DiscordException=Exception,
        Color=SimpleNamespace(green=lambda: 0, red=lambda: 0),
        app_commands=SimpleNamespace(
            CommandTree=DummyCommandTree,
            describe=lambda *a, **k: (lambda f: f),
        ),
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


@pytest.fixture(autouse=True)
def reset_moderation_rate_limits(discord_module: object) -> None:
    from src.interfaces import discord_moderation

    discord_moderation._ACTION_COUNTS.clear()
    discord_moderation._COOLDOWNS.clear()


class DummyInteraction:
    def __init__(self) -> None:
        self.response = SimpleNamespace(send_message=AsyncMock())
        self.channel = None
        self.user = SimpleNamespace(
            id="user-1", guild_permissions=SimpleNamespace(administrator=True)
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_start_broadcasts_success_embed(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_start_embed=MagicMock(return_value={}),
        channel_to_agent={},
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(discord_module, "start_simulation", AsyncMock())
    interaction = DummyInteraction()
    await discord_module.slash_start(interaction)
    bot.create_start_embed.assert_called_once_with(True)
    bot.send_simulation_update.assert_awaited_once_with(embed={})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_start_broadcasts_failure_embed(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_start_embed=MagicMock(return_value={}),
        channel_to_agent={},
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(
        discord_module, "start_simulation", AsyncMock(side_effect=RuntimeError("boom"))
    )
    interaction = DummyInteraction()
    await discord_module.slash_start(interaction)
    bot.create_start_embed.assert_called_once()
    args = bot.create_start_embed.call_args[0]
    assert args[0] is False and "boom" in args[1]
    bot.send_simulation_update.assert_awaited_once_with(embed={})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_stop_broadcasts_success_embed(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_stop_embed=MagicMock(return_value={}),
        channel_to_agent={},
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(discord_module, "stop_simulation", AsyncMock())
    interaction = DummyInteraction()
    await discord_module.slash_stop(interaction)
    bot.create_stop_embed.assert_called_once_with(True)
    bot.send_simulation_update.assert_awaited_once_with(embed={})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_spawn_broadcasts_success_embed(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_spawn_embed=MagicMock(return_value={}),
        channel_to_agent={},
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(discord_module, "spawn_agent_command", AsyncMock())
    interaction = DummyInteraction()
    await discord_module.slash_spawn(interaction, "agent")
    bot.create_spawn_embed.assert_called_once_with("agent", True)
    bot.send_simulation_update.assert_awaited_once_with(embed={})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_spawn_broadcasts_failure_embed(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_spawn_embed=MagicMock(return_value={}),
        channel_to_agent={},
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(
        discord_module, "spawn_agent_command", AsyncMock(side_effect=RuntimeError("bad"))
    )
    interaction = DummyInteraction()
    await discord_module.slash_spawn(interaction, "agent")
    bot.create_spawn_embed.assert_called_once()
    args = bot.create_spawn_embed.call_args[0]
    assert args[0] == "agent" and args[1] is False
    bot.send_simulation_update.assert_awaited_once_with(embed={})


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
    await sim.external_event_ingestion.handle_human_command("/kb first")
    await sim.external_event_ingestion.handle_human_command("/kb second")
    sim.knowledge_board.add_entry.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_message_relay_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.sim.simulation as simulation_module
    from src.infra import ledger as ledger_module
    from src.sim.simulation import Simulation

    class DummyAgent:
        def get_id(self) -> str:
            return "a"

        agent_id = "a"
        state = SimpleNamespace(ip=10.0, du=10.0)

    sim = Simulation([])
    sim.agents = [DummyAgent()]
    monkeypatch.setattr(
        simulation_module,
        "get_resource_manager",
        lambda: SimpleNamespace(ensure_du_budget=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(ledger_module.ledger, "spend", AsyncMock())
    await sim.external_event_ingestion.handle_human_command("hello")
    await sim.external_event_ingestion.handle_human_command("hello again")
    ledger_module.ledger.spend.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_event_enqueues_control_event(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ctx = SimpleNamespace(get_event_queue=lambda: queue)
    bot = SimpleNamespace(context=ctx)
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(
        discord_module, "_has_control_command_permission", AsyncMock(return_value=True)
    )
    interaction = DummyInteraction()
    interaction.user = SimpleNamespace(
        id="u-1", guild_permissions=SimpleNamespace(administrator=False)
    )
    await discord_module.slash_event(interaction, "meteor shower")
    event = await queue.get()
    assert event.type == "control"
    assert event.data["command"] == "inject_event"
    assert event.data["text"] == "meteor shower"
    assert event.data["scope"] == "global"
    assert event.data["author"] == str(interaction.user)
    interaction.response.send_message.assert_awaited_once_with("event injected", ephemeral=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_event_requires_authorization(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ctx = SimpleNamespace(get_event_queue=lambda: queue)
    bot = SimpleNamespace(context=ctx)
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    monkeypatch.setattr(
        discord_module, "_has_control_command_permission", AsyncMock(return_value=False)
    )
    interaction = DummyInteraction()
    await discord_module.slash_event(interaction, "storm")
    assert queue.empty()
    interaction.response.send_message.assert_awaited_once_with("unauthorized", ephemeral=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_nudge_enqueues_event(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    ctx = SimpleNamespace(get_event_queue=lambda: queue)
    bot = SimpleNamespace(context=ctx)
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    interaction = DummyInteraction()
    await discord_module.slash_nudge(interaction, "hi there")
    event = await queue.get()
    assert event.type == "nudge" and event.data == {"prompt": "hi there"}
    interaction.response.send_message.assert_awaited_once_with("nudge sent", ephemeral=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_spawn_forwards_optional_profile_fields(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = SimpleNamespace(
        context=discord_module.DEFAULT_CONTEXT,
        send_simulation_update=AsyncMock(),
        create_spawn_embed=MagicMock(return_value={}),
        channel_to_agent={},
    )
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    spawn_mock = AsyncMock()
    monkeypatch.setattr(discord_module, "spawn_agent_command", spawn_mock)

    interaction = DummyInteraction()
    await discord_module.slash_spawn(
        interaction,
        "agent",
        role="Analyzer",
        persona="critical thinker",
        backstory="former reviewer",
        traits_json='{"openness": 0.4}',
        empathy=0.7,
    )

    spawn_mock.assert_awaited_once()
    args = spawn_mock.await_args
    assert args.args[:2] == ("agent", bot.context)
    assert args.kwargs["role"] == "Analyzer"
    assert args.kwargs["persona"] == "critical thinker"
    assert args.kwargs["backstory"] == "former reviewer"
    assert args.kwargs["traits"] == {"openness": 0.4, "empathy": 0.7}
