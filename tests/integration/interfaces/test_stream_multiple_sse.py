import asyncio
import json
from collections.abc import AsyncGenerator

import pytest

pytest.importorskip("fastapi")
import fastapi

if not hasattr(fastapi.FastAPI(), "__call__"):
    pytest.skip("FastAPI ASGI app not available", allow_module_level=True)

import pytest

from src import http_app
from src.interfaces import dashboard_backend as db
from src.sim import event_bus


class SimpleESR:
    def __init__(self, gen: AsyncGenerator[dict[str, str], None]) -> None:
        self.gen = gen


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_events_multiple_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_app, "EventSourceResponse", SimpleESR)
    bus = event_bus.EventBus()
    monkeypatch.setattr(event_bus, "get_event_bus", lambda: bus)

    async def publish() -> None:
        await bus.publish(db.SimulationEvent(type="one", data={"step": 1}))
        await bus.publish(db.SimulationEvent(type="two", data={"step": 2}))
        bus.shutdown()

    publisher = asyncio.create_task(publish())
    resp = await http_app.stream_events(DummyRequest())

    events = []
    for _ in range(2):
        evt = await resp.gen.__anext__()
        events.append(evt)

    await publisher

    assert [e["event"] for e in events] == ["simulation_event", "simulation_event"]
    data = [json.loads(e["data"]) for e in events]
    assert [d["step"] for d in data] == [1, 2]


class DummyRequest:
    def __init__(self) -> None:
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_messages_multiple_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "EventSourceResponse", SimpleESR)
    queue: asyncio.Queue[db.AgentMessage] = asyncio.Queue()
    monkeypatch.setattr(db, "message_sse_queue", queue)

    msg1 = db.AgentMessage(agent_id="a", content="one", step=1)
    msg2 = db.AgentMessage(agent_id="b", content="two", step=2)
    await db.enqueue_message(msg1)
    await db.enqueue_message(msg2)

    resp = await db.stream_messages(DummyRequest())

    events = [await resp.gen.__anext__() for _ in range(2)]

    contents = [json.loads(e["data"])["content"] for e in events]
    assert contents == ["one", "two"]
