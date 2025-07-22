from __future__ import annotations

import asyncio

# Skip self argument annotation warnings in stub classes
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

from src.governance.law_board import law_board
from src.governance.service import governance
from src.infra.ledger import ledger
from src.sim.event_bus import get_event_bus

from .widget_registry import WidgetRegistry

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
else:  # pragma: no cover - optional runtime dependency
    try:
        from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse
    except Exception:

        class FastAPI:
            def __init__(self: FastAPI, *args: object, **kwargs: object) -> None:
                pass

            def get(self: FastAPI, *args: object, **kwargs: object) -> Callable[[Any], Any]:
                def dec(fn: Any) -> Any:
                    return fn

                return dec

            def post(self: FastAPI, *args: object, **kwargs: object) -> Callable[[Any], Any]:
                def dec(fn: Any) -> Any:
                    return fn

                return dec

            def websocket(self: FastAPI, *args: object, **kwargs: object) -> Callable[[Any], Any]:
                def dec(fn: Any) -> Any:
                    return fn

                return dec

        class Request:  # pragma: no cover - minimal stub
            pass

        class Response:  # pragma: no cover - minimal stub
            def __init__(self: Response, *args: object, **kwargs: object) -> None:
                pass

        class WebSocket:  # pragma: no cover - minimal stub
            pass

        class WebSocketDisconnect(Exception):
            pass

        class JSONResponse:  # pragma: no cover - minimal stub
            def __init__(
                self: JSONResponse, content: object, *args: object, **kwargs: object
            ) -> None:
                self.body = json.dumps(content).encode("utf-8")


if TYPE_CHECKING:
    from sse_starlette.sse import EventSourceResponse
else:  # pragma: no cover - optional dependency
    try:
        from sse_starlette.sse import EventSourceResponse
    except Exception:

        class EventSourceResponse:  # pragma: no cover - minimal stub
            def __init__(self: EventSourceResponse, *args: object, **kwargs: object) -> None:
                self.gen = None


# Global queue for agent messages with bounded size
message_sse_queue: asyncio.Queue[AgentMessage] = asyncio.Queue(maxsize=1000)


async def enqueue_message(msg: AgentMessage) -> None:
    """Add a message to the SSE queue, dropping the oldest if full."""
    if message_sse_queue.full():
        try:
            message_sse_queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - unlikely
            pass
    await message_sse_queue.put(msg)


# Queue for general simulation events streamed via SSE/WebSocket. Calls to
# :func:`get_event_queue` return a queue bound to the current event loop and
# subscribed to the global :class:`~src.sim.event_bus.EventBus` instance. The
# same queue is returned on subsequent calls within the same loop to match the
# previous behaviour.
_event_queue: asyncio.Queue[SimulationEvent | None] | None = None
_event_queue_loop: asyncio.AbstractEventLoop | None = None


def get_event_queue() -> asyncio.Queue[SimulationEvent | None]:
    """Return a shared event queue bound to the active loop."""
    global _event_queue, _event_queue_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no running loop
        loop = asyncio.new_event_loop()
    if _event_queue is None or _event_queue_loop is not loop:
        if _event_queue is not None:
            get_event_bus().unsubscribe(_event_queue)
        _event_queue = get_event_bus().subscribe()
        _event_queue_loop = loop
    return _event_queue


# Simulation control state
SIM_STATE: dict[str, Any] = {
    "paused": False,
    "speed": 1.0,
    "semantic_manager": None,
    "simulation": None,
}
BREAKPOINT_TAGS: set[str] = {"violence", "nsfw"}

# Registry of widgets registered by the UI or plugins
WIDGET_REGISTRY = WidgetRegistry()

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

    type: str
    data: dict[str, Any] | None = None


class LawProposal(BaseModel):
    proposer_id: str
    text: str


class Proposal(BaseModel):
    """Proposal with optional vote weights."""

    proposer_id: str
    text: str
    vote_weights: dict[str, int] | None = None


class LawsResponse(BaseModel):
    laws: list[str]


class VoteRecord(BaseModel):
    proposer_id: str
    text: str
    approved: bool
    yes_weight: float
    no_weight: float
    ts: Any


class VotesResponse(BaseModel):
    votes: list[VoteRecord]


app = FastAPI()


@app.get(
    "/stream/messages",
    response_class=EventSourceResponse,
    response_model=None,
)
async def stream_messages(request: Request) -> Response:
    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        while True:
            if await request.is_disconnected():
                break
            try:
                msg: AgentMessage = await message_sse_queue.get()
                yield {
                    "event": "message",
                    "data": msg.model_dump_json(),
                }
            except (RuntimeError, ValueError) as e:
                yield {"event": "error", "data": json.dumps({"error": str(e)})}

    generator: AsyncGenerator[dict[str, Any], None] = event_generator()
    return EventSourceResponse(generator)  # type: ignore[no-any-return]


@app.get(
    "/api/map",
    response_class=EventSourceResponse,
    response_model=None,
)
async def api_map(request: Request) -> Response:
    """Stream map_change events as Server-Sent Events."""

    bus = get_event_bus()
    queue = bus.subscribe()

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                event: SimulationEvent | None = await queue.get()
                if event is None:
                    break
                if event.type == "map_change":
                    yield {"data": event.model_dump_json()}
        finally:
            bus.unsubscribe(queue)

    generator: AsyncGenerator[dict[str, Any], None] = event_generator()
    return EventSourceResponse(generator)  # type: ignore[no-any-return]


@app.get("/health")
async def health() -> Response:
    return JSONResponse({"status": "ok"})


@app.get("/api/missions")
async def get_missions() -> Response:
    """Return the list of missions from the bundled JSON file."""
    with open(MISSIONS_PATH, encoding="utf-8") as f:
        missions = json.load(f)
    return JSONResponse(missions)


@app.get("/api/quests")
async def get_quests_api() -> Response:
    """Return the list of generated quests."""
    quests = ledger.get_quests()
    return JSONResponse({"quests": quests})


@app.get("/api/agents/{agent_id}/semantic_summaries")
async def get_semantic_summaries(agent_id: str, limit: int = 3) -> Response:
    """Return recent semantic summaries for an agent."""
    manager = SIM_STATE.get("semantic_manager")
    if manager is not None:
        try:
            summaries = manager.get_semantic_summaries(agent_id, limit=limit)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to get semantic summaries for %s", agent_id, exc_info=exc)
            return JSONResponse({"error": "summary retrieval failed"}, status_code=500)
    else:
        summaries = []
    return JSONResponse({"summaries": summaries})


@app.post("/api/propose_law")
async def api_propose_law(proposal: LawProposal) -> Response:
    """Submit a law proposal to the active simulation."""
    sim = SIM_STATE.get("simulation")
    approved = False
    if sim is not None:
        try:
            approved = await sim.propose_law(proposal.proposer_id, proposal.text)
        except Exception:  # pragma: no cover - defensive
            approved = False
    return JSONResponse({"approved": approved})


@app.post("/api/propose")
async def api_propose(proposal: Proposal) -> Response:
    """Submit a weighted law proposal to the active simulation."""
    sim = SIM_STATE.get("simulation")
    approved = False
    if sim is not None:
        proposer = next((a for a in sim.agents if a.agent_id == proposal.proposer_id), None)
        if proposer is not None:
            try:
                approved = await governance.propose_law(
                    proposer,
                    proposal.text,
                    sim.agents,
                    proposal.vote_weights,
                )
            except Exception:  # pragma: no cover - defensive
                approved = False
    return JSONResponse({"approved": approved})


@app.get("/api/proposals")
async def api_get_proposals(limit: int = 10) -> Response:
    """Return stored law proposals."""
    proposals = governance.get_proposals(limit)
    return JSONResponse({"proposals": proposals})


@app.get("/api/laws", response_model=LawsResponse)
async def api_get_laws() -> Response:
    """Return passed laws from the law board."""

    laws = law_board.get_laws()
    return JSONResponse({"laws": laws})


@app.get("/api/votes", response_model=VotesResponse)
async def api_get_votes(limit: int = 10) -> Response:
    """Return stored voting history."""

    votes = ledger.get_law_proposals(limit)
    return JSONResponse({"votes": votes})


@app.get("/api/token_balances")
async def api_token_balances() -> Response:
    """Return DU/IP and token balances for all agents."""

    def _load() -> dict[str, dict[str, Any]]:
        cur = ledger.conn.execute("SELECT agent_id, ip, du FROM agent_balances")
        balances: dict[str, dict[str, Any]] = {
            row[0]: {"ip": float(row[1]), "du": float(row[2]), "tokens": {}}
            for row in cur.fetchall()
        }
        cur = ledger.conn.execute("SELECT agent_id, token, amount FROM agent_tokens")
        for agent_id, token, amount in cur.fetchall():
            agent = balances.setdefault(agent_id, {"ip": 0.0, "du": 0.0, "tokens": {}})
            agent["tokens"][token] = int(amount)
        return balances

    agents = await asyncio.to_thread(_load)
    return JSONResponse({"agents": agents})


@app.get("/api/auctions")
async def api_auctions() -> Response:
    """Return auctions and their current bids."""

    def _load() -> list[dict[str, Any]]:
        cur = ledger.conn.execute("SELECT id, item, status, winner_id FROM auctions")
        auctions: list[dict[str, Any]] = []
        for row in cur.fetchall():
            bid_rows = ledger.conn.execute(
                "SELECT agent_id, amount FROM bids WHERE auction_id=? ORDER BY amount DESC, id ASC",
                (row[0],),
            ).fetchall()
            auctions.append(
                {
                    "id": int(row[0]),
                    "item": row[1],
                    "status": row[2],
                    "winner_id": row[3],
                    "bids": [{"agent_id": b[0], "amount": float(b[1])} for b in bid_rows],
                }
            )
        return auctions

    auctions = await asyncio.to_thread(_load)
    return JSONResponse({"auctions": auctions})


async def register_widget(widget: dict[str, Any]) -> Response:
    """Register a widget provided by the UI or a plugin."""
    name = widget.get("name")
    if isinstance(name, str):
        meta = {k: v for k, v in widget.items() if k != "name"}
        WIDGET_REGISTRY.register(name, meta)
    return JSONResponse({"widgets": WIDGET_REGISTRY.list()})


@app.post("/api/register_widget")
async def register_widget_legacy(widget: dict[str, Any]) -> Response:
    """Backward compatible widget registration endpoint."""
    return await register_widget(widget)


try:
    app.post("/api/register_widget")(register_widget)
except AttributeError:  # pragma: no cover - stub app may lack decorators
    pass


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    await websocket.accept()
    bus = get_event_bus()
    queue = bus.subscribe()
    try:
        while True:
            event: SimulationEvent | None = await queue.get()
            if event is None:
                break
            await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(queue)
        await websocket.close()


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
    async def control(request: Request) -> Response:
        body = await request.body()
        try:
            command = json.loads(body)
            if not isinstance(command, dict):
                raise ValueError("payload must be a JSON object")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Invalid control payload: %s", exc)
            return JSONResponse({"error": "invalid"})

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
                    if not isinstance(cmd, dict):
                        raise ValueError("payload must be a JSON object")
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    logger.warning("Invalid control payload via WS: %s", exc)
                    await websocket.send_text(json.dumps({"error": "invalid"}))
                    continue
                result = await handle_control_command(cmd)
                await websocket.send_text(json.dumps(result))
        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()

except AttributeError:  # pragma: no cover - stub app may lack decorators
    pass


async def emit_event(event: SimulationEvent) -> None:
    """Emit a simulation event and check for breakpoints."""
    bus = get_event_bus()
    await bus.publish(event)
    tags = set(event.data.get("tags", [])) if event.data else set()
    if tags & BREAKPOINT_TAGS:
        SIM_STATE["paused"] = True
        await bus.publish(
            SimulationEvent(
                type="breakpoint_hit",
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
            type="map_action",
            data={"agent_id": agent_id, "step": step, "action": action, **details},
        )
    )


async def emit_map_change_event(world_map: dict[str, Any]) -> None:
    """Convenience helper to enqueue world map updates."""
    await emit_event(SimulationEvent(type="map_change", data={"world_map": world_map}))


__all__ = [
    "WIDGET_REGISTRY",
    "EventSourceResponse",
    "LawProposal",
    "LawsResponse",
    "Proposal",
    "SimulationEvent",
    "VoteRecord",
    "VotesResponse",
    "api_auctions",
    "api_get_laws",
    "api_get_proposals",
    "api_get_votes",
    "api_map",
    "api_propose_law",
    "api_token_balances",
    "app",
    "emit_event",
    "emit_map_action_event",
    "emit_map_change_event",
    "enqueue_message",
    "get_event_bus",
    "get_event_queue",
    "get_quests_api",
    "message_sse_queue",
    "register_widget",
]
