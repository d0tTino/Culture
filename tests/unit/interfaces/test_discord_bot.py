import asyncio
import importlib
import sys
from types import SimpleNamespace
from typing import Callable
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
    class DummyCommandTree:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def command(self, *a: object, **k: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
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
        Color=SimpleNamespace(blue=lambda: 0, green=lambda: 0, gold=lambda: 0, purple=lambda: 0, red=lambda: 0),
        app_commands=SimpleNamespace(
            CommandTree=DummyCommandTree,
            describe=lambda *a, **k: (lambda f: f),
        ),
    )
    dummy_app = SimpleNamespace(
        spawn_agent_command=AsyncMock(),
        start_simulation=AsyncMock(),
        stop_simulation=AsyncMock(),
    )
    dummy_db = SimpleNamespace(
        DEFAULT_CONTEXT=SimpleNamespace(
            sim_state={}, _event_queue=None, _event_queue_loop=None, message_queue=asyncio.Queue()
        ),
        AgentMessage=SimpleNamespace,
        SimulationEvent=SimpleNamespace,
        message_sse_queue=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "src.app", dummy_app)
    monkeypatch.setitem(
        sys.modules, "src.infra.config", SimpleNamespace(get_config=lambda *a, **k: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "src.infra.ledger",
        SimpleNamespace(
            ledger=SimpleNamespace(get_balance_async=AsyncMock()),
            log_penalty=lambda *a, **k: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "src.interfaces.dashboard_backend", dummy_db)
    monkeypatch.setitem(
        sys.modules,
        "src.interfaces.metrics",
        SimpleNamespace(get_llm_latency=lambda: 0, get_kb_size=lambda: 0),
    )
    monkeypatch.setitem(
        sys.modules, "src.sim.context", SimpleNamespace(SimulationContext=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules,
        "src.utils.policy",
        SimpleNamespace(allow_message=lambda *a, **k: True, evaluate_with_opa=lambda c: (True, c)),
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
    assert isinstance(discord_module.SimulationDiscordBot.create_step_start_embed(bot, 1), dict)
    assert isinstance(discord_module.SimulationDiscordBot.create_step_end_embed(bot, 2), dict)
    assert isinstance(
        discord_module.SimulationDiscordBot.create_knowledge_board_embed(bot, "a", "msg", 3),
        dict,
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
def test_notify_budget_exceeded_without_running_loop(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_payloads: list[object] = []

    class DummyBot:
        def __init__(self) -> None:
            self.embeds: list[dict[str, object] | None] = []

        async def send_simulation_update(self, *, embed: dict[str, object] | None = None) -> None:
            self.embeds.append(embed)

    dummy_bot = DummyBot()

    monkeypatch.setattr(discord_module, "message_sse_queue", SimpleNamespace(put_nowait=captured_payloads.append))
    monkeypatch.setattr(discord_module, "get_active_bot", lambda: dummy_bot)
    monkeypatch.setattr(
        discord_module, "metrics", SimpleNamespace(get_llm_latency_p95=lambda: 42.0)
    )
    monkeypatch.setattr(
        discord_module.ledger, "get_balance", lambda agent_id: (1.0, 2.0), raising=False
    )

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        discord_module.notify_budget_exceeded("agent-123", required=10.0, remaining=-1.0)
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    assert captured_payloads, "SSE queue should receive at least one payload"
    payload = captured_payloads[0]
    assert getattr(payload, "agent_id", None) == "agent-123"
    assert getattr(payload, "extra", {}).get("required") == 10.0
    assert getattr(payload, "extra", {}).get("remaining") == -1.0
    assert dummy_bot.embeds, "Fallback path should dispatch embed synchronously"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_forward_agent_messages_embed(
    discord_module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyEmbed:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.kwargs = kwargs

        def set_author(self, *args: object, **kwargs: object) -> None:
            pass

        def add_field(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(discord_module, "discord", SimpleNamespace(Embed=DummyEmbed))
    bot = object.__new__(discord_module.SimulationDiscordBot)
    bot.is_ready = True

    async def fake_send_simulation_update(**kwargs: object) -> None:
        fake_send_simulation_update.kwargs = kwargs

    fake_send_simulation_update.kwargs = None

    bot.send_simulation_update = fake_send_simulation_update  # type: ignore
    msg = discord_module.AgentMessage(
        agent_id="agent12345678",
        content="ignored",
        step=1,
        extra={"embed": {"title": "t", "description": "d", "color": 0x1}},
    )
    embed_payload = msg.extra["embed"]
    await bot.send_simulation_update(
        content=None, embed=embed_payload, agent_id=msg.agent_id, recipient=None
    )
    assert fake_send_simulation_update.kwargs is not None
    assert fake_send_simulation_update.kwargs["embed"] is not None
    assert fake_send_simulation_update.kwargs["content"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_span_emission(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import src.interfaces.discord_bot as discord_module

    class MockSpan:
        def __init__(self, name: str) -> None:
            self.name = name
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

        def __enter__(self) -> "MockSpan":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    class MockTracer:
        def __init__(self) -> None:
            self.spans: list[MockSpan] = []

        def start_as_current_span(self, name: str) -> MockSpan:
            span = MockSpan(name)
            self.spans.append(span)
            return span

    tracer = MockTracer()
    monkeypatch.setattr(discord_module, "tracer", tracer)
    monkeypatch.setattr(discord_module, "allow_message", lambda c: True)
    monkeypatch.setattr(discord_module, "evaluate_with_opa", AsyncMock(return_value=(True, "hi")))
    monkeypatch.setattr(
        discord_module.ledger, "get_balance_async", AsyncMock(return_value=(1.0, 1.0))
    )
    monkeypatch.setattr(discord_module.metrics.HUMAN_MESSAGES_TOTAL, "inc", lambda: None)
    monkeypatch.setattr(discord_module.config, "get_config", lambda k: None)
    monkeypatch.setattr(discord_module, "send_interaction_response", AsyncMock())

    class DummyClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.user = SimpleNamespace(id=0)
            self.handlers: dict[str, object] = {}

        def event(self, fn: object) -> object:
            self.handlers[getattr(fn, "__name__", "")] = fn
            return fn

        def get_channel(self, channel_id: int) -> SimpleNamespace:
            return SimpleNamespace(id=channel_id, send=AsyncMock())

    monkeypatch.setattr(discord_module.discord, "Client", DummyClient)

    bot = await discord_module.SimulationDiscordBot.create(
        "token", 123, context=discord_module.DEFAULT_CONTEXT
    )
    bot.is_ready = True
    bot.channel_to_agent[123] = "agent1"
    bot.event_queue = asyncio.Queue()
    message = SimpleNamespace(
        author=SimpleNamespace(id=7),
        channel=SimpleNamespace(id=123, send=AsyncMock()),
        content="hello",
    )
    await bot.clients["token"].handlers["on_message"](message)

    span = tracer.spans[0]
    assert span.name == "discord.message"
    assert span.attributes["discord.channel.id"] == 123
    assert span.attributes["discord.user.id"] == 7
    assert span.attributes["discord.agent.id"] == "agent1"
    assert "discord.latency_ms" in span.attributes

    tracer.spans.clear()
    monkeypatch.setattr(discord_module, "get_active_bot", lambda ctx=None: bot)
    interaction = SimpleNamespace(
        channel=SimpleNamespace(id=123),
        user=SimpleNamespace(id=7),
        response=SimpleNamespace(send_message=AsyncMock()),
    )
    await discord_module.slash_status(interaction)

    span = tracer.spans[0]
    assert span.name == "discord.command"
    assert span.attributes["discord.command.name"] == "status"
    assert span.attributes["discord.channel.id"] == 123
    assert span.attributes["discord.user.id"] == 7
    assert span.attributes["discord.agent.id"] == "agent1"
    assert "discord.latency_ms" in span.attributes
