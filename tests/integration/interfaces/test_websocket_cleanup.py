import asyncio
import json

import pytest

from src.interfaces import dashboard_backend as db


class DummyBus:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.queue: asyncio.Queue[db.SimulationEvent | None] = asyncio.Queue()

    def subscribe(self) -> asyncio.Queue[db.SimulationEvent | None]:
        return self.queue

    def unsubscribe(self, q: asyncio.Queue[db.SimulationEvent | None]) -> None:
        assert q is self.queue
        self.events.append("unsubscribe")


class DummyWebSocket:
    def __init__(self, events: list[str], receive_exc: Exception | None = None) -> None:
        self.events = events
        self.accepted = False
        self.closed = False
        self.receive_exc = receive_exc

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, text: str) -> None:
        # Not used in these tests
        json.loads(text)

    async def receive_text(self) -> str:
        if self.receive_exc is not None:
            raise self.receive_exc
        return "{}"

    async def close(self) -> None:
        self.events.append("close")
        self.closed = True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_events_unsubscribe_before_close(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    bus = DummyBus(events)
    await bus.queue.put(None)
    monkeypatch.setattr(db, "get_event_bus", lambda: bus)

    ws = DummyWebSocket(events)
    await db.websocket_events(ws)

    assert events == ["unsubscribe", "close"]
    assert ws.closed is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_control_closes_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    ws = DummyWebSocket(events, receive_exc=db.WebSocketDisconnect())
    await db.ws_control(ws)
    assert events == ["close"]
    assert ws.closed is True
