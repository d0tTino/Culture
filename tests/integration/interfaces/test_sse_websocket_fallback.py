import asyncio
import socket

import httpx
import pytest
import uvicorn
import websockets

from src import http_app
from src.interfaces import dashboard_backend as db
from src.sim import event_bus


class FailingESR:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")


async def _start_server() -> tuple[uvicorn.Server, asyncio.Task[None], int]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()
    config = uvicorn.Config(
        http_app.app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        loop="asyncio",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task, port


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sse_websocket_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = event_bus.EventBus()
    monkeypatch.setattr(http_app, "get_event_bus", lambda: bus)
    monkeypatch.setattr(db, "get_event_bus", lambda: bus)
    monkeypatch.setattr(http_app, "EventSourceResponse", FailingESR)

    server, task, port = await _start_server()

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"http://127.0.0.1:{port}/stream/events")
    assert resp.status_code == 500

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/events") as ws:
        await bus.publish(db.SimulationEvent(type="tick", data={"step": 1}))
        bus.shutdown()
        data = await asyncio.wait_for(ws.recv(), 1)
        event = db.SimulationEvent.model_validate_json(data)
        assert event.data is not None
        assert event.data["step"] == 1
        with pytest.raises(websockets.ConnectionClosedOK):
            await ws.recv()

    server.should_exit = True
    await task
