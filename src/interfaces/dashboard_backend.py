import asyncio

# Skip self argument annotation warnings in stub classes
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
else:  # pragma: no cover - optional runtime dependency
    try:
        from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse
    except Exception:

        class FastAPI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get(self, *args: object, **kwargs: object) -> Callable[[Any], Any]:
                def dec(fn: Any) -> Any:
                    return fn

                return dec

            def post(self, *args: object, **kwargs: object) -> Callable[[Any], Any]:
                def dec(fn: Any) -> Any:
                    return fn

                return dec

            def websocket(self, *args: object, **kwargs: object) -> Callable[[Any], Any]:
                def dec(fn: Any) -> Any:
                    return fn

                return dec

        class Request:  # pragma: no cover - minimal stub
            pass

        class Response:  # pragma: no cover - minimal stub
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        class WebSocket:  # pragma: no cover - minimal stub
            pass

        class WebSocketDisconnect(Exception):
            pass

        class JSONResponse:  # pragma: no cover - minimal stub
            def __init__(self, content: object, *args: object, **kwargs: object) -> None:
                self.body = json.dumps(content).encode("utf-8")


from pydantic import BaseModel

if TYPE_CHECKING:
    from sse_starlette.sse import EventSourceResponse
else:  # pragma: no cover - optional dependency
    try:
        from sse_starlette.sse import EventSourceResponse
    except Exception:

        class EventSourceResponse:  # pragma: no cover - minimal stub
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.gen = None


# Global queue for agent messages
message_sse_queue: asyncio.Queue["AgentMessage"] = asyncio.Queue()

# Queue for general simulation events streamed via SSE/WebSocket.  The queue is
# lazily created so it binds to the currently running event loop, avoiding
# cross-loop issues in tests.
_event_queue: asyncio.Queue["SimulationEvent | None"] | None = None


def get_event_queue() -> asyncio.Queue["SimulationEvent | None"]:
    """Return the shared event queue, creating it for the active loop."""
    global _event_queue
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no running loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if _event_queue is None or _event_queue._loop is not loop:
        _event_queue = asyncio.Queue()
    return _event_queue

# Simulation control state
SIM_STATE: dict[str, Any] = {"paused": False, "speed": 1.0, "semantic_manager": None}
BREAKPOINT_TAGS: set[str] = {"violence", "nsfw"}

# Path to the initial missions data bundled with the front-end
MISSIONS_PATH = (
    Path(__file__).resolve().parents[2] / "culture-ui" / "src" / "mock" / "missions.json"
)


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
async def stream_messages(request: Request) -> EventSourceResponse:
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


@app.get("/api/missions")
async def get_missions() -> Response:
    """Return the list of missions from the bundled JSON file."""
    with open(MISSIONS_PATH, encoding="utf-8") as f:
        missions = json.load(f)
    return JSONResponse(missions)


@app.get("/api/agents/{agent_id}/semantic_summaries")
async def get_semantic_summaries(agent_id: str, limit: int = 3) -> Response:
    """Return recent semantic summaries for an agent."""
    manager = SIM_STATE.get("semantic_manager")
    summaries: list[str] = []
    if manager is not None:
        try:
            summaries = manager.get_recent_summaries(agent_id, limit=limit)
        except Exception:  # pragma: no cover - defensive
            summaries = []
    return JSONResponse({"summaries": summaries})


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = get_event_queue()
    try:
        while True:
            event: SimulationEvent | None = await queue.get()
            if event is None:
                break
            await websocket.send_text(event.json())
    except WebSocketDisconnect:
        pass


async def handle_control_command(cmd: dict[str, Any]) -> dict[str, Any]:
    """Process a control command and update simulation state."""
    action = cmd.get("command")
    if action == "pause":
        SIM_STATE["paused"] = True
    elif action == "resume":
        SIM_STATE["paused"] = False
    elif action == "set_speed":
        try:
            SIM_STATE["speed"] = float(cmd.get("value", 1))
        except (TypeError, ValueError):
            pass
    elif action == "set_breakpoints":
        tags = cmd.get("tags")
        if isinstance(tags, list):
            BREAKPOINT_TAGS.clear()
            BREAKPOINT_TAGS.update(str(t) for t in tags)
    return {**SIM_STATE, "breakpoints": list(BREAKPOINT_TAGS)}


try:

    @app.post("/control")
    async def control(command: dict[str, Any]) -> Response:
        result = await handle_control_command(command)
        return JSONResponse(result)

except AttributeError:  # pragma: no cover - stub app may lack decorators
    pass


try:

    @app.websocket("/ws/control")
    async def ws_control(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    cmd = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"error": "invalid"}))
                    continue
                result = await handle_control_command(cmd)
                await websocket.send_text(json.dumps(result))
        except WebSocketDisconnect:
            pass

except AttributeError:  # pragma: no cover - stub app may lack decorators
    pass


async def emit_event(event: SimulationEvent) -> None:
    """Emit a simulation event and check for breakpoints."""
    queue = get_event_queue()
    await queue.put(event)
    tags = set(event.data.get("tags", [])) if event.data else set()
    if tags & BREAKPOINT_TAGS:
        SIM_STATE["paused"] = True
        await queue.put(
            SimulationEvent(
                event_type="breakpoint_hit",
                data={
                    "tags": list(tags & BREAKPOINT_TAGS),
                    "step": event.data.get("step") if event.data else None,
                },
            )
        )


async def emit_map_action_event(
    agent_id: str,
    step: int,
    action: str,
    **details: Any,
) -> None:
    """Convenience helper to enqueue map actions."""
    await emit_event(
        SimulationEvent(
            event_type="map_action",
            data={"agent_id": agent_id, "step": step, "action": action, **details},
        )
    )


__all__ = [
    "EventSourceResponse",
    "SimulationEvent",
    "app",
    "emit_event",
    "emit_map_action_event",
    "get_event_queue",
    "message_sse_queue",
]
