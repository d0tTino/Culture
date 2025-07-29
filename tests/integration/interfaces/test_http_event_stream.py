import asyncio
from collections.abc import AsyncGenerator

import httpx
import pytest

pytest.importorskip("fastapi")
import fastapi

if not hasattr(fastapi.FastAPI(), "__call__"):
    pytest.skip("FastAPI ASGI app not available", allow_module_level=True)

from src import http_app
from src.interfaces import dashboard_backend as db
from src.sim import event_bus


class SimpleESR:
    def __init__(self, gen: AsyncGenerator[dict[str, str], None]) -> None:
        self.gen = gen


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_event_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Events published on the bus should stream over HTTP in order."""
    monkeypatch.setattr(http_app, "EventSourceResponse", SimpleESR)
    bus = event_bus.EventBus()
    monkeypatch.setattr(event_bus, "get_event_bus", lambda: bus)
    monkeypatch.setattr(db, "get_event_bus", lambda: bus)

    transport = httpx.ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def _read() -> list[db.SimulationEvent]:
            events: list[db.SimulationEvent] = []
            async with client.stream("GET", "/stream/events") as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line.removeprefix("data: ")
                        events.append(db.SimulationEvent.model_validate_json(payload))
                        if len(events) == 3:
                            break
            return events

        reader = asyncio.create_task(_read())
        await asyncio.sleep(0.05)
        await bus.publish(db.SimulationEvent(type="one", data={"step": 1}))
        await bus.publish(db.SimulationEvent(type="two", data={"step": 2}))
        await bus.publish(db.SimulationEvent(type="three", data={"step": 3}))
        bus.shutdown()
        events = await reader

    assert [e.type for e in events] == ["one", "two", "three"]
    assert [e.data["step"] for e in events if e.data] == [1, 2, 3]
