import json

import pytest

from src.interfaces import dashboard_backend as db


class DummyRequest:
    async def is_disconnected(self) -> bool:
        return False


class DummyWS:
    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


async def _clear_event_queue() -> None:
    queue = db.get_event_queue()
    while not queue.empty():
        _ = await queue.get()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_action_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.http_app as http_app

    class CaptureESR:
        def __init__(self, gen: object) -> None:
            self.gen = gen

    monkeypatch.setattr(http_app, "EventSourceResponse", CaptureESR)
    await _clear_event_queue()
    await db.emit_map_action_event("A", 1, "move", position=(1, 0))
    queue = db.get_event_queue()
    await queue.put(None)
    resp = await http_app.stream_events(DummyRequest())
    event = await resp.gen.__anext__()
    data = json.loads(event["data"])
    assert data["event_type"] == "map_action"
    assert data["data"]["agent_id"] == "A"
    with pytest.raises(StopAsyncIteration):
        await resp.gen.__anext__()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_action_websocket() -> None:
    ws = DummyWS()
    await _clear_event_queue()
    await db.emit_map_action_event("B", 2, "gather", resource="wood", success=True)
    queue = db.get_event_queue()
    await queue.put(None)
    await db.websocket_events(ws)
    payload = json.loads(ws.sent[0])
    assert payload["event_type"] == "map_action"
    assert payload["data"]["agent_id"] == "B"
