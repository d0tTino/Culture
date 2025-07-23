import asyncio
import socket
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest

httpx_ws = pytest.importorskip("httpx_ws")
import uvicorn
import websockets

from src.sim import event_bus

pytest.importorskip("fastapi")
import fastapi

if not hasattr(fastapi.FastAPI(), "__call__"):
    pytest.skip("FastAPI ASGI app not available", allow_module_level=True)

from src import http_app
from src.interfaces import dashboard_backend as db
from src.sim.simulation import Simulation


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = SimpleNamespace(ip=0.0, du=0.0)

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: SimpleNamespace) -> None:
        self.state = new_state

    async def run_turn(
        self,
        simulation_step: int,
        environment_perception: dict[str, Any] | None = None,
        memory_service: object | None = None,
        vector_store_manager: object | None = None,
        knowledge_board: object | None = None,
    ) -> dict[str, Any]:
        return {"step": simulation_step}


async def _clear_event_queue() -> None:
    queue = db.get_event_queue()
    while not queue.empty():
        await queue.get()


from collections.abc import AsyncGenerator


class SimpleESR:
    def __init__(self, gen: AsyncGenerator[dict[str, str], None]) -> None:
        self.gen = gen

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[[dict[str, Any]], Any],
    ) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        async for event in self.gen:
            data = f"event: {event['event']}\ndata: {event['data']}\n\n".encode()
            await send({"type": "http.response.body", "body": data, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_events_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_app, "EventSourceResponse", SimpleESR)

    async def _allow(_: str) -> bool:
        return True

    monkeypatch.setattr("src.governance.policy.evaluate_policy", _allow)
    monkeypatch.setattr("src.sim.simulation.evaluate_policy", _allow)
    await _clear_event_queue()
    agent = DummyAgent("agent1")
    sim = Simulation([agent])  # type: ignore[list-item]

    await sim.run_step()

    queue = db.get_event_queue()
    await queue.put(None)

    transport = httpx.ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("GET", "/stream/events") as resp:
            data_line = None
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_line = line.removeprefix("data: ")
                    break
    assert data_line is not None
    event = db.SimulationEvent.model_validate_json(data_line)
    assert event.data is not None
    assert event.data["agent_id"] == "agent1"


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
async def test_ws_events_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(http_app, "EventSourceResponse", SimpleESR)

    async def _allow(_: str) -> bool:
        return True

    monkeypatch.setattr("src.governance.policy.evaluate_policy", _allow)
    monkeypatch.setattr("src.sim.simulation.evaluate_policy", _allow)
    await _clear_event_queue()
    agent = DummyAgent("agent1")
    sim = Simulation([agent])  # type: ignore[list-item]

    await sim.run_step()

    queue = db.get_event_queue()
    await queue.put(None)

    server, task, port = await _start_server()
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/events") as ws:
        data = await asyncio.wait_for(ws.recv(), 1)
    event = db.SimulationEvent.model_validate_json(data)
    assert event.data is not None
    assert event.data["agent_id"] == "agent1"
    server.should_exit = True
    await task


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stream_events_emit_event_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = event_bus.EventBus()
    monkeypatch.setattr(http_app, "get_event_bus", lambda: bus)
    monkeypatch.setattr(db, "get_event_bus", lambda: bus)

    transport = httpx.ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def _read() -> str | None:
            async with client.stream("GET", "/stream/events") as resp:
                data_line = None
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_line = line.removeprefix("data: ")
                        break
                return data_line

        read_task = asyncio.create_task(_read())
        await asyncio.sleep(0.05)
        await db.emit_event(db.SimulationEvent(type="tick", data={"step": 1}))
        bus.shutdown()
        data_line = await read_task

    assert data_line is not None
    event = db.SimulationEvent.model_validate_json(data_line)
    assert event.data is not None
    assert event.data["step"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ws_events_emit_event_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = event_bus.EventBus()
    monkeypatch.setattr(http_app, "get_event_bus", lambda: bus)
    monkeypatch.setattr(db, "get_event_bus", lambda: bus)

    transport = httpx_ws.transport.ASGIWebSocketTransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with httpx_ws.aconnect_ws("ws://test/ws/events", client) as ws:
            await db.emit_event(db.SimulationEvent(type="tick", data={"step": 2}))
            bus.shutdown()
            data = await ws.receive_text()

    event = db.SimulationEvent.model_validate_json(data)
    assert event.data is not None
    assert event.data["step"] == 2
