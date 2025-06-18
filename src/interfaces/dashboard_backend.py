import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Global queue for agent messages
message_sse_queue: asyncio.Queue["AgentMessage"] = asyncio.Queue()
# Queue for general simulation events streamed via SSE/WebSocket
event_queue: asyncio.Queue["SimulationEvent | None"] = asyncio.Queue()


class AgentMessage(BaseModel):
    agent_id: str
    content: str
    step: int
    recipient_id: str | None = None
    action_intent: str | None = None
    timestamp: float | None = None
    extra: dict[str, Any] | None = None


class SimulationEvent(BaseModel):
    """Generic simulation event structure for dashboards."""

    event_type: str
    data: dict[str, Any] | None = None


app = FastAPI()


@app.get("/stream/messages")
async def stream_messages(request: Request) -> EventSourceResponse:  # type: ignore[no-any-unimported]
    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        while True:
            if await request.is_disconnected():
                break
            try:
                msg: AgentMessage = await message_sse_queue.get()
                yield {
                    "event": "message",
                    "data": msg.json(),
                }
            except (RuntimeError, ValueError) as e:
                yield {"event": "error", "data": json.dumps({"error": str(e)})}

    generator: AsyncGenerator[dict[str, Any], None] = event_generator()
    return EventSourceResponse(generator)


@app.get("/health")
async def health() -> Response:
    return JSONResponse({"status": "ok"})


@app.get("/stream/events")
async def stream_events(request: Request) -> EventSourceResponse:  # type: ignore[no-any-unimported]
    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        while True:
            if await request.is_disconnected():
                break
            event: SimulationEvent | None = await event_queue.get()
            if event is None:
                break
            yield {"event": "simulation_event", "data": event.json()}

    return EventSourceResponse(event_generator())


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            event: SimulationEvent | None = await event_queue.get()
            if event is None:
                break
            await websocket.send_text(event.json())
    except WebSocketDisconnect:
        pass
