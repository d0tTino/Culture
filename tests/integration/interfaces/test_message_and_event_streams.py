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

    await db.emit_event(db.SimulationEvent(type="update", data={"step": 3}))
    await event_queue.put(None)

    ws = DummyWebSocket()
    await db.websocket_events(ws)
    assert ws.accepted is True
    payload = json.loads(ws.sent[0])
    assert payload["data"]["step"] == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_sse_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: asyncio.Queue[db.AgentMessage] = asyncio.Queue()
    monkeypatch.setattr(db, "message_sse_queue", queue)

    class CaptureESR:
        def __init__(self, gen: object) -> None:
            self.gen = gen

    monkeypatch.setattr(db, "EventSourceResponse", CaptureESR)

    msg1 = db.AgentMessage(agent_id="a", content="one", step=1)
    msg2 = db.AgentMessage(agent_id="b", content="two", step=2)
    await db.enqueue_message(msg1)
    await db.enqueue_message(msg2)

    resp1 = await db.stream_messages(DummyRequest())
    resp2 = await db.stream_messages(DummyRequest())

    event1, event2 = await asyncio.gather(
        resp1.gen.__anext__(),
        resp2.gen.__anext__(),
    )

    contents = {json.loads(event1["data"])["content"], json.loads(event2["data"])["content"]}
    assert contents == {"one", "two"}
    assert queue.empty()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_messages_error(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: asyncio.Queue[db.AgentMessage] = asyncio.Queue()
    monkeypatch.setattr(db, "message_sse_queue", queue)

    class CaptureESR:
        def __init__(self, gen: object) -> None:
            self.gen = gen

    monkeypatch.setattr(db, "EventSourceResponse", CaptureESR)

    class BadMessage(db.AgentMessage):
        def json(self, *args: object, **kwargs: object) -> str:  # type: ignore[override]
            raise ValueError("boom")

    await queue.put(BadMessage(agent_id="bad", content="x", step=1))

    resp = await db.stream_messages(DummyRequest())
    event = await resp.gen.__anext__()
    assert event["event"] == "error"
    assert json.loads(event["data"])["error"] == "boom"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    event_queue: asyncio.Queue[db.SimulationEvent | None] = asyncio.Queue()
    monkeypatch.setattr(db, "get_event_queue", lambda: event_queue)

    await event_queue.put(db.SimulationEvent(type="one", data={"step": 1}))
    await event_queue.put(db.SimulationEvent(type="two", data={"step": 2}))
    await event_queue.put(None)

    class DisconnectingWebSocket(DummyWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        async def send_text(self, text: str) -> None:
            await super().send_text(text)
            self.count += 1
            if self.count == 1:
                raise db.WebSocketDisconnect()

    ws1 = DisconnectingWebSocket()
    await db.websocket_events(ws1)
    assert len(ws1.sent) == 1

    ws2 = DummyWebSocket()
    await db.websocket_events(ws2)
    assert [json.loads(t)["type"] for t in ws2.sent] == ["two"]
