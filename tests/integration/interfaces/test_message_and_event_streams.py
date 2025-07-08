import asyncio
import json

import pytest

from src.interfaces import dashboard_backend as db


class DummyRequest:
    def __init__(self) -> None:
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > 1


class DummyWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_messages_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: asyncio.Queue[db.AgentMessage] = asyncio.Queue()
    monkeypatch.setattr(db, "message_sse_queue", queue)

    class CaptureESR:
        def __init__(self, gen: object) -> None:
            self.gen = gen

    monkeypatch.setattr(db, "EventSourceResponse", CaptureESR)

    msg = db.AgentMessage(agent_id="a", content="hello", step=1)
    await db.enqueue_message(msg)

    resp = await db.stream_messages(DummyRequest())
    event = await resp.gen.__anext__()
    payload = json.loads(event["data"])
    assert payload["content"] == "hello"
    with pytest.raises(StopAsyncIteration):
        await resp.gen.__anext__()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_events_deliver_simulation_events(monkeypatch: pytest.MonkeyPatch) -> None:
    event_queue: asyncio.Queue[db.SimulationEvent | None] = asyncio.Queue()
    monkeypatch.setattr(db, "get_event_queue", lambda: event_queue)

    await db.emit_event(db.SimulationEvent(event_type="update", data={"step": 3}))
    await event_queue.put(None)

    ws = DummyWebSocket()
    await db.websocket_events(ws)
    assert ws.accepted is True
    payload = json.loads(ws.sent[0])
    assert payload["data"]["step"] == 3
