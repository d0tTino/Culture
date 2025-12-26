import asyncio
import json

import pytest

from src.interfaces import dashboard_backend as db
from src.sim.event_bus import get_event_bus


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

    async def close(self) -> None:
        return None


async def _reset_event_bus() -> None:
    bus = get_event_bus()
    bus.shutdown()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_action_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.http_app as http_app

    class CaptureESR:
        def __init__(self, gen: object) -> None:
            self.gen = gen

    monkeypatch.setattr(http_app, "EventSourceResponse", CaptureESR)
    await _reset_event_bus()
    resp = await http_app.stream_events(DummyRequest())
    next_event = asyncio.create_task(resp.gen.__anext__())
    await asyncio.sleep(0)
    await db.emit_map_action_event("A", 1, "move", position=(1, 0))
    event = await next_event
    data = json.loads(event["data"])
    assert data["type"] == "map_action"
    assert data["data"]["agent_id"] == "A"
    get_event_bus().shutdown()
    with pytest.raises(StopAsyncIteration):
        await resp.gen.__anext__()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_map_action_websocket() -> None:
    ws = DummyWS()
    await _reset_event_bus()
    task = asyncio.create_task(db.websocket_events(ws))
    await asyncio.sleep(0)
    await db.emit_map_action_event("B", 2, "gather", resource="wood", success=True)
    get_event_bus().shutdown()
    await task
    payload = json.loads(ws.sent[0])
    assert payload["type"] == "map_action"
    assert payload["data"]["agent_id"] == "B"
