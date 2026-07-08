import asyncio
import json
import socket

import pytest
import uvicorn
import websockets
from websockets import exceptions as ws_exceptions

from src import http_app
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


@pytest.fixture(autouse=True)
def _reset_api_token() -> None:
    db.configure_api_token(None)
    yield
    db.configure_api_token(None)


async def _start_server() -> tuple[uvicorn.Server, asyncio.Task[None], int]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _host, port = sock.getsockname()
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
async def test_ws_events_receive_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    bus = DummyBus(events)
    await bus.queue.put(db.SimulationEvent(type="tick", data={"step": 1}))
    await bus.queue.put(None)
    monkeypatch.setattr(db, "get_event_bus", lambda: bus)

    server, task, port = await _start_server()
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/events") as ws:
        data = await asyncio.wait_for(ws.recv(), 1)
        event = db.SimulationEvent.model_validate_json(data)
        assert event.data is not None
        assert event.data["step"] == 1
        with pytest.raises(websockets.ConnectionClosedOK):
            await ws.recv()
    server.should_exit = True
    await task
    assert events == ["unsubscribe"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_control_pause_and_disconnect() -> None:
    server, task, port = await _start_server()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws/control") as ws:
            await ws.send(json.dumps({"command": "pause"}))
            data = await asyncio.wait_for(ws.recv(), 1)
            payload = json.loads(data)
            assert payload["paused"] is True
            await ws.close()
            with pytest.raises(websockets.ConnectionClosedOK):
                await ws.recv()
    finally:
        server.should_exit = True
        await task


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_control_rejects_missing_token() -> None:
    db.configure_api_token("secret")
    server, task, port = await _start_server()
    try:
        with pytest.raises(ws_exceptions.InvalidStatus):
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/control"):
                pass
    finally:
        server.should_exit = True
        await task


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_control_rejects_invalid_token() -> None:
    db.configure_api_token("secret")
    server, task, port = await _start_server()
    try:
        with pytest.raises(ws_exceptions.InvalidStatus):
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/control?token=wrong"):
                pass
    finally:
        server.should_exit = True
        await task


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_control_accepts_valid_token_and_processes_command() -> None:
    db.configure_api_token("secret")
    server, task, port = await _start_server()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws/control?token=secret") as ws:
            await ws.send(json.dumps({"command": "pause"}))
            data = await asyncio.wait_for(ws.recv(), 1)
            payload = json.loads(data)
            assert payload["paused"] is True
    finally:
        server.should_exit = True
        await task
