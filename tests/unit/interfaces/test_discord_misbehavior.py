import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class DummyContext:
    def __init__(self) -> None:
        self.sim_state: dict[str, object] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_queue_loop = None
        self.message_queue: asyncio.Queue = asyncio.Queue()

    def get_event_queue(self) -> asyncio.Queue:
        return self._event_queue


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

        def command(self, *a: object, **k: object):
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
        Color=SimpleNamespace(),
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

    ctx = DummyContext()
    dummy_db = SimpleNamespace(
        DEFAULT_CONTEXT=ctx,
        AgentMessage=SimpleNamespace,
        SimulationEvent=SimpleNamespace,
        message_sse_queue=SimpleNamespace(),
        SNAPSHOT_DIR=Path("/tmp"),
    )

    store_calls: list[tuple[int, int, Path | None]] = []
    log_calls: list[dict[str, object]] = []
    fetch_events_list: list[dict[str, object]] = []

    def store_replay_slice(start: int, end: int, directory: Path | None = None) -> Path:
        store_calls.append((start, end, directory))
        directory = directory or Path("/tmp")
        return directory / f"replay_{start}_{end}.jsonl"

    def log_misbehavior(event: dict[str, object]) -> dict[str, object]:
        log_calls.append(event)
        return event

    def fetch_events(event_type: str | None = None):
        return list(fetch_events_list)

    dummy_event_log = SimpleNamespace(
        store_replay_slice=store_replay_slice,
        log_misbehavior=log_misbehavior,
        fetch_events=fetch_events,
    )

    monkeypatch.setitem(sys.modules, "src.app", dummy_app)
    monkeypatch.setitem(
        sys.modules, "src.infra.config", SimpleNamespace(get_config=lambda *a, **k: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "src.infra.ledger",
        SimpleNamespace(
            ledger=SimpleNamespace(get_balance_async=AsyncMock()), log_penalty=lambda *a, **k: None
        ),
    )
    monkeypatch.setitem(sys.modules, "src.interfaces.dashboard_backend", dummy_db)
    monkeypatch.setitem(
        sys.modules,
        "src.interfaces.metrics",
        SimpleNamespace(get_llm_latency=lambda: 0, get_kb_size=lambda: 0),
    )
    monkeypatch.setitem(
        sys.modules, "src.sim.context", SimpleNamespace(SimulationContext=DummyContext)
    )
    monkeypatch.setitem(
        sys.modules,
        "src.utils.policy",
        SimpleNamespace(allow_message=lambda *a, **k: True, evaluate_with_opa=lambda c: (True, c)),
    )
    monkeypatch.setitem(sys.modules, "src.infra.event_log", dummy_event_log)
    monkeypatch.setitem(sys.modules, "discord", dummy_discord)
    monkeypatch.setitem(sys.modules, "discord.app_commands", dummy_discord.app_commands)
    monkeypatch.setitem(sys.modules, "discord.ext", SimpleNamespace(commands=dummy_commands))
    monkeypatch.setitem(sys.modules, "discord.ext.commands", dummy_commands)

    module = importlib.reload(importlib.import_module("src.interfaces.discord_bot"))
    return module, ctx, store_calls, log_calls, fetch_events_list


@pytest.fixture()
def discord_module(monkeypatch: pytest.MonkeyPatch):
    module, ctx, store_calls, log_calls, fetch_events_list = reload_module(monkeypatch)
    return module, ctx, store_calls, log_calls, fetch_events_list


@pytest.mark.unit
@pytest.mark.asyncio
async def test_record_misbehavior(discord_module):
    module, ctx, store_calls, log_calls, _ = discord_module
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    await module.record_misbehavior(interaction, "agent1", "bad")
    assert store_calls == [(0, 0, module.SNAPSHOT_DIR)]
    assert log_calls[0]["agent_id"] == "agent1"
    evt = ctx._event_queue.get_nowait()
    assert evt.type == "misbehavior"
    assert evt.data["reason"] == "bad"
    interaction.response.send_message.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slash_misbehavior_log(discord_module):
    module, ctx, store_calls, log_calls, fetch_events_list = discord_module
    fetch_events_list.extend(
        [
            {"step": 1, "agent_id": "a", "reason": "bad", "replay_path": "p1"},
            {"step": 2, "agent_id": "b", "reason": "worse", "replay_path": "p2"},
        ]
    )
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))
    await module.slash_misbehavior_log(interaction, limit=1)
    interaction.response.send_message.assert_awaited_once()
    sent = interaction.response.send_message.call_args[0][0]
    assert "worse" in sent and "p2" in sent
