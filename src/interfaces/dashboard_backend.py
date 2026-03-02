from __future__ import annotations

import asyncio

# Skip self argument annotation warnings in stub classes
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from opentelemetry import trace
from pydantic import BaseModel

from src.governance.decision_kernel import PolicyDecisionService
from src.governance.rules_engine import governance_rules_engine
from src.governance.service import governance
from src.infra import event_log
from src.infra import metrics as infra_metrics
from src.infra.ledger import ledger
from src.interfaces import metrics
from src.interfaces.domain_command_adapters import command_from_payload
from src.sim.context import SimulationContext
from src.sim.event_bus import get_event_bus
from src.sim.persistence.snapshot_service import SnapshotPersistenceService

from .widget_registry import WidgetRegistry

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "snapshots"

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# JSON response body for semantic summary retrieval errors
SEMANTIC_SUMMARIES_ERROR: Final[dict[str, str]] = {"error": "summary retrieval failed"}

# Default simulation context used by module-level APIs
DEFAULT_CONTEXT = SimulationContext()

API_TOKEN: str | None = None
DECISION_KERNEL = PolicyDecisionService()


def configure_api_token(token: str | None) -> None:
    """Configure the token required for state-changing requests."""
    global API_TOKEN
    API_TOKEN = token


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

            def middleware(self: FastAPI, *args: object, **kwargs: object) -> Callable[[Any], Any]:
                def dec(fn: Any) -> Any:
                    return fn

                return dec

        class Request:  # pragma: no cover - minimal stub
            def __init__(
                self: Request,
                headers: dict[str, str] | None = None,
                method: str = "GET",
            ) -> None:
                self.headers = headers or {}
                self.method = method

        class Response:  # pragma: no cover - minimal stub
            def __init__(self: Response, *args: object, **kwargs: object) -> None:
                self.status_code = kwargs.get("status_code", 200)
                self.body = kwargs.get("content", b"")

        class WebSocket:  # pragma: no cover - minimal stub
            pass

        class WebSocketDisconnect(Exception):
            pass

        class JSONResponse:  # pragma: no cover - minimal stub
            def __init__(
                self: JSONResponse, content: object, *args: object, **kwargs: object
            ) -> None:
                self.body = json.dumps(content).encode("utf-8")
                self.status_code = kwargs.get("status_code", 200)


if TYPE_CHECKING:
    from sse_starlette.sse import EventSourceResponse
else:  # pragma: no cover - optional dependency
    try:
        from sse_starlette.sse import EventSourceResponse
    except Exception:

        class EventSourceResponse:  # pragma: no cover - minimal stub
            def __init__(self: EventSourceResponse, *args: object, **kwargs: object) -> None:
                self.gen = None


# Agent message queue stored in the simulation context
message_sse_queue = DEFAULT_CONTEXT.message_queue


async def enqueue_message(msg: AgentMessage, ctx: SimulationContext = DEFAULT_CONTEXT) -> None:
    """Add a message to the SSE queue, dropping the oldest if full."""
    queue = ctx.message_queue
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - unlikely
            pass
    await queue.put(msg)


# Queue for general simulation events is managed by the context


def get_event_queue(
    ctx: SimulationContext = DEFAULT_CONTEXT,
) -> asyncio.Queue[SimulationEvent | None]:
    """Return a shared event queue bound to the active loop."""
    return cast(asyncio.Queue[SimulationEvent | None], ctx.get_event_queue())


# Simulation control state
SIM_STATE = DEFAULT_CONTEXT.sim_state
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


def board_payload_to_embed(payload: dict[str, Any]) -> dict[str, Any]:
    """Map a knowledge board payload to Discord embed fields."""
    agent_id = str(payload.get("agent_id", ""))
    content = str(payload.get("content", ""))
    step = int(payload.get("step", 0))
    return {
        "title": f"📝 New Knowledge Board Entry (Step {step})",
        "description": f"```{content}```",
        "color": 0xFFD700,
        "author": {"name": f"Posted by Agent {agent_id[:8]}"},
    }


class SimulationEvent(BaseModel):
    """Generic simulation event structure for dashboards."""

    type: str
    data: dict[str, Any] | None = None


class LawProposal(BaseModel):
    proposer_id: str
    text: str
    vote_weights: dict[str, int] | None = None


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


class ProposalRecord(VoteRecord):
    ip_spent: float


class VotesResponse(BaseModel):
    votes: list[VoteRecord]


class ProposalsResponse(BaseModel):
    proposals: list[ProposalRecord]


class GovernanceReadModelResponse(BaseModel):
    rules: list[dict[str, Any]]
    current_rules: list[dict[str, Any]] = []
    pending_votes: list[dict[str, Any]] = []
    active_offices: list[dict[str, Any]] = []
    sanctions: list[dict[str, Any]] = []


class VoteRequest(BaseModel):
    """Vote payload for manual voting."""

    agent_id: str
    text: str
    approve: bool
    weight: int | None = 1


class StakeRequest(BaseModel):
    """Request payload for staking influence points."""

    agent_id: str
    amount: float


app = FastAPI()


async def _require_token(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Enforce bearer token for mutating requests when configured."""
    if API_TOKEN and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        auth = request.headers.get("Authorization")
        if auth != f"Bearer {API_TOKEN}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


if hasattr(app, "middleware"):
    app.middleware("http")(_require_token)
else:  # pragma: no cover - support older FastAPI versions
    try:  # pragma: no cover - optional dependency
        from starlette.middleware.base import BaseHTTPMiddleware

        app.add_middleware(BaseHTTPMiddleware, dispatch=_require_token)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive
        logger.warning("API token enforcement disabled: middleware unsupported")


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
    return cast(Response, EventSourceResponse(generator))


@app.get(
    "/api/map/stream",
    response_class=EventSourceResponse,
    response_model=None,
)
async def stream_map(request: Request) -> Response:
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
    return cast(Response, EventSourceResponse(generator))


@app.get(
    "/stream/agents",
    response_class=EventSourceResponse,
    response_model=None,
)
async def stream_agents(request: Request) -> Response:
    """Stream agent state, mood, and top memories."""

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        while True:
            if await request.is_disconnected():
                break
            sim = SIM_STATE.get("simulation")
            agents: list[dict[str, Any]] = []
            if sim is not None:
                memory_service = getattr(sim, "memory_service", None)
                for ag in sim.agents:
                    try:
                        state = cast(dict[str, Any], ag.state.model_dump())
                    except Exception:  # pragma: no cover - defensive
                        state = {}
                    mood = getattr(ag.state, "mood_value", None)
                    memories: list[str] = []
                    if memory_service is not None:
                        try:
                            memories = memory_service.get_recent_semantic_summaries(
                                ag.agent_id, limit=3
                            )
                        except Exception:  # pragma: no cover - defensive
                            memories = []
                    agents.append(
                        {
                            "agent_id": ag.agent_id,
                            "state": state,
                            "mood": mood,
                            "memories": memories,
                        }
                    )
            yield {"data": json.dumps({"agents": agents})}
            await asyncio.sleep(1)

    generator: AsyncGenerator[dict[str, Any], None] = event_generator()
    return cast(Response, EventSourceResponse(generator))


try:

    @app.websocket("/ws/agents")
    async def ws_agents(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                sim = SIM_STATE.get("simulation")
                agents: list[dict[str, Any]] = []
                if sim is not None:
                    memory_service = getattr(sim, "memory_service", None)
                    for ag in sim.agents:
                        try:
                            state = cast(dict[str, Any], ag.state.model_dump())
                        except Exception:  # pragma: no cover - defensive
                            state = {}
                        mood = getattr(ag.state, "mood_value", None)
                        memories: list[str] = []
                        if memory_service is not None:
                            try:
                                memories = memory_service.get_recent_semantic_summaries(
                                    ag.agent_id, limit=3
                                )
                            except Exception:  # pragma: no cover - defensive
                                memories = []
                        agents.append(
                            {
                                "agent_id": ag.agent_id,
                                "state": state,
                                "mood": mood,
                                "memories": memories,
                            }
                        )
                await websocket.send_text(json.dumps({"agents": agents}))
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()

except AttributeError:  # pragma: no cover - stub app may lack decorators
    pass


@app.get("/api/map")
async def api_map() -> Response:
    """Return the latest world map state with agent mood and summaries."""
    sim = SIM_STATE.get("simulation")
    world_map = sim.world_map.to_dict() if sim is not None else {}
    agents: dict[str, dict[str, Any]] = {}
    if sim is not None:
        memory_service = getattr(sim, "memory_service", None)
        for ag in sim.agents:
            mood = getattr(ag.state, "mood_value", None)
            summary = ""
            if memory_service is not None:
                try:
                    summaries = memory_service.get_recent_semantic_summaries(ag.agent_id, limit=1)
                    if summaries:
                        summary = summaries[0]
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Failed to load summary for %s", ag.agent_id)
            agents[ag.agent_id] = {"mood": mood, "summary": summary}
    return JSONResponse({"world_map": world_map, "agents": agents})


@app.get("/api/agent_stats")
async def api_agent_stats() -> Response:
    """Return retrieval counts, mood, and cost metrics for each agent."""
    sim = SIM_STATE.get("simulation")
    stats: dict[str, dict[str, Any]] = {}
    retrieval_counts: dict[str, int] = {}
    if sim is not None:
        store = getattr(getattr(sim, "memory_service", None), "vector_store", None)

        if store is not None:

            def _load() -> dict[str, int]:
                counts: dict[str, int] = {}
                try:
                    results = store.collection.get(include=["metadatas"])
                    metadatas = results.get("metadatas") or []
                    for meta in metadatas:
                        if not meta:
                            continue
                        agent_id = meta.get("agent_id")
                        if agent_id is None:
                            continue
                        counts[agent_id] = counts.get(agent_id, 0) + int(
                            meta.get("retrieval_count", 0)
                        )
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Failed to gather retrieval stats")
                return counts

            retrieval_counts = await asyncio.to_thread(_load)

        for ag in sim.agents:
            mood = getattr(ag.state, "mood_value", None)
            du_per_1k = infra_metrics.get_agent_du_per_1k_tokens(ag.agent_id)
            try:
                llm_latency_p95 = float(
                    metrics.AGENT_LLM_LATENCY_P95_MS.labels(agent_id=ag.agent_id)._value.get()  # type: ignore[attr-defined]
                )
            except Exception:  # pragma: no cover - defensive
                llm_latency_p95 = 0.0
            stats[ag.agent_id] = {
                "mood": mood,
                "retrieval_count": retrieval_counts.get(ag.agent_id, 0),
                "du_per_1k_tokens": du_per_1k,
                "llm_latency_p95_ms": llm_latency_p95,
            }
    return JSONResponse({"agents": stats})


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
    manager = DEFAULT_CONTEXT.sim_state.get("semantic_manager")
    if manager is not None:
        try:
            summaries = manager.get_semantic_summaries(agent_id, limit=limit)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to get semantic summaries for %s", agent_id, exc_info=exc)
            return JSONResponse(SEMANTIC_SUMMARIES_ERROR, status_code=500)
    else:
        summaries = []
    return JSONResponse({"summaries": summaries})


@app.get("/api/agents/{agent_id}/state")
async def get_agent_state(agent_id: str) -> Response:
    """Return the current state for an agent."""
    sim = DEFAULT_CONTEXT.sim_state.get("simulation")
    state: dict[str, Any] | None = None
    if sim is not None:
        agent = next((a for a in sim.agents if a.agent_id == agent_id), None)
        if agent is not None:
            try:
                state = cast(dict[str, Any], agent.state.model_dump())
            except Exception:  # pragma: no cover - defensive
                state = {}
    return JSONResponse({"state": state or {}})


@app.get("/api/agents/{agent_id}/memories")
async def get_agent_memories(agent_id: str, limit: int = 5) -> Response:
    """Return recent raw memories for an agent."""
    sim = DEFAULT_CONTEXT.sim_state.get("simulation")
    memories: list[dict[str, Any]] = []
    if sim is not None:
        memory_service = getattr(sim, "memory_service", None)
        vector_store = getattr(memory_service, "vector_store", None)
        if vector_store is not None:
            try:
                memories = vector_store.retrieve_filtered_memories(agent_id, limit=limit)
            except Exception:  # pragma: no cover - defensive
                memories = []
    return JSONResponse({"memories": memories})


@app.get("/api/memory/{agent_id}")
async def api_memory(agent_id: str, limit: int = 5) -> Response:
    """Return semantic and episodic memories for an agent."""
    semantic: list[str] = []
    episodic: list[dict[str, Any]] = []

    manager = DEFAULT_CONTEXT.sim_state.get("semantic_manager")
    if manager is not None:
        try:
            semantic = manager.get_semantic_summaries(agent_id, limit=limit)
        except Exception:  # pragma: no cover - defensive
            semantic = []

    sim = DEFAULT_CONTEXT.sim_state.get("simulation")
    if sim is not None:
        memory_service = getattr(sim, "memory_service", None)
        vector_store = getattr(memory_service, "vector_store", None)
        if vector_store is not None:
            try:
                episodic = vector_store.retrieve_filtered_memories(agent_id, limit=limit)
            except Exception:  # pragma: no cover - defensive
                episodic = []

    return JSONResponse({"semantic": semantic, "episodic": episodic})


@app.post("/api/propose_law")
async def api_propose_law(proposal: LawProposal) -> Response:
    """Submit a (weighted) law proposal to the active simulation."""
    sim = DEFAULT_CONTEXT.sim_state.get("simulation")
    approved = False
    if sim is not None:
        try:
            if proposal.vote_weights is None:
                approved = await sim.propose_law(
                    proposal.proposer_id,
                    proposal.text,
                )
            else:
                approved = await sim.propose_law(
                    proposal.proposer_id,
                    proposal.text,
                    proposal.vote_weights,
                )
        except Exception:  # pragma: no cover - defensive
            approved = False
    return JSONResponse({"approved": approved})


@app.post("/api/governance/propose")
async def api_governance_propose(proposal: Proposal) -> Response:
    """Submit a proposal via the governance service and return the outcome."""
    sim = DEFAULT_CONTEXT.sim_state.get("simulation")
    result: dict[str, float | bool] = {
        "approved": False,
        "yes_weight": 0.0,
        "no_weight": 0.0,
        "ip_spent": 0.0,
    }
    if sim is not None:
        proposer = next((a for a in sim.agents if a.agent_id == proposal.proposer_id), None)
        if proposer is not None:
            try:
                result = await governance.propose_law(
                    proposer,
                    proposal.text,
                    sim.agents,
                    proposal.vote_weights,
                )
            except Exception:  # pragma: no cover - defensive
                result["approved"] = False
    return JSONResponse(result)


@app.post("/api/vote")
async def api_vote(vote: VoteRequest) -> Response:
    """Submit a manual vote for a proposal."""
    sim = DEFAULT_CONTEXT.sim_state.get("simulation")
    result = False
    if sim is not None:
        agent = next((a for a in sim.agents if a.agent_id == vote.agent_id), None)
        if agent is not None:
            try:
                result = await governance.vote_weighted(
                    agent, vote.text, vote.weight or 1, vote.approve
                )
            except Exception:  # pragma: no cover - defensive
                result = False
    return JSONResponse({"vote": result})


@app.post("/api/stake_ip")
async def api_stake_ip(req: StakeRequest) -> Response:
    """Stake influence points for an agent and return the total staked."""
    try:
        total = await governance.stake_ip(req.agent_id, req.amount)
    except Exception:  # pragma: no cover - defensive
        total = 0.0
    return JSONResponse({"staked_ip": total})


@app.get("/api/proposals", response_model=ProposalsResponse)
async def api_get_proposals(limit: int = 10) -> Response:
    """Return stored law proposals."""
    proposals = governance.get_proposals(limit)
    return JSONResponse({"proposals": proposals})


@app.get("/api/recent_proposals", response_model=ProposalsResponse)
async def api_recent_proposals(limit: int = 5) -> Response:
    """Return the most recent law proposals."""
    proposals = governance.get_proposals(limit)
    return JSONResponse({"proposals": proposals})


@app.get("/gov", response_model=GovernanceReadModelResponse)
@app.get("/api/gov", response_model=GovernanceReadModelResponse)
async def api_get_gov() -> Response:
    """Return active governance rules and enforcement stats."""
    sim = SIM_STATE.get("simulation")
    if sim is not None and hasattr(sim, "get_governance_read_model"):
        model = cast(dict[str, Any], sim.get_governance_read_model())
        return JSONResponse(model)
    return JSONResponse(
        {
            "rules": governance_rules_engine.current_rules(),
            "current_rules": governance_rules_engine.current_rules(),
            "pending_votes": governance_rules_engine.pending_votes(),
            "active_offices": governance_rules_engine.active_offices(),
            "sanctions": governance_rules_engine.sanctions(),
        }
    )


@app.get("/api/laws", response_model=LawsResponse)
async def api_get_laws() -> Response:
    """Return passed laws from the canonical governance state store."""

    return JSONResponse({"laws": governance_rules_engine.passed_laws()})


@app.get("/api/gov/current_rules")
async def api_current_rules() -> Response:
    """Return current executable governance rules."""
    return JSONResponse({"current_rules": governance_rules_engine.current_rules()})


@app.get("/api/gov/pending_votes")
async def api_pending_votes() -> Response:
    """Return pending governance vote records."""
    return JSONResponse({"pending_votes": governance_rules_engine.pending_votes()})


@app.get("/api/gov/active_offices")
async def api_active_offices() -> Response:
    """Return currently active governance offices."""
    return JSONResponse({"active_offices": governance_rules_engine.active_offices()})


@app.get("/api/gov/sanctions")
async def api_sanctions() -> Response:
    """Return active sanctions and post-action enforcement records."""
    return JSONResponse({"sanctions": governance_rules_engine.sanctions()})


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
        for agent_id, agent in balances.items():
            agent["remaining_du_budget"] = infra_metrics.get_agent_du_budget(agent_id)
            agent["llm_latency_p95_ms"] = infra_metrics.get_agent_llm_latency_p95(agent_id)
        return balances

    agents = await asyncio.to_thread(_load)
    return JSONResponse({"agents": agents})


def _cost_metrics_data() -> dict[str, float | int]:
    """Collect DU cost, sentiment, and reliability metrics for dashboards."""

    memory_retrievals = metrics.get_memory_retrievals()
    memory_retrieval_errors = metrics.get_memory_retrieval_errors()
    memory_retrieval_attempts = memory_retrievals + memory_retrieval_errors
    memory_success_rate = (
        memory_retrievals / memory_retrieval_attempts if memory_retrieval_attempts else 0.0
    )
    memory_error_rate = (
        memory_retrieval_errors / memory_retrieval_attempts if memory_retrieval_attempts else 0.0
    )

    llm_errors_total = metrics.get_llm_errors_total()
    llm_calls_total = metrics.get_llm_calls_total()
    llm_error_rate = llm_errors_total / llm_calls_total if llm_calls_total else 0.0

    return {
        "du_per_1k_tokens": metrics.get_du_per_1k_tokens(),
        "llm_latency_p95_ms": metrics.get_llm_latency_p95(),
        "coalition_count": metrics.get_coalition_count(),
        "average_sentiment": metrics.get_average_sentiment(),
        "rag_hit_rate": metrics.get_rag_hit_rate(),
        "memory_retrievals_total": memory_retrievals,
        "memory_retrieval_errors_total": memory_retrieval_errors,
        "memory_retrieval_success_rate": memory_success_rate,
        "memory_retrieval_error_rate": memory_error_rate,
        "llm_errors_total": llm_errors_total,
        "llm_error_rate": llm_error_rate,
    }


@app.get("/api/cost_metrics")
async def api_cost_metrics() -> Response:
    """Return DU cost and latency metrics for dashboards."""

    return JSONResponse(_cost_metrics_data())


@app.get("/api/observability_metrics")
async def api_observability_metrics() -> Response:
    """Return DU cost and latency metrics for dashboards."""

    return JSONResponse(_cost_metrics_data())


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


@app.get("/api/memory_snapshots")
async def api_memory_snapshots(limit: int = 10) -> Response:
    """List available memory snapshot steps."""

    steps = sorted(
        int(p.stem.split("_")[1])
        for p in SNAPSHOT_DIR.glob("snapshot_*.json*")
        if p.stem.split("_")[1].isdigit()
    )
    resp = JSONResponse({"steps": steps[-limit:]})
    if not hasattr(resp, "status_code"):
        resp.status_code = 200
    return resp


@app.get("/api/memory_snapshots/{step}")
async def api_memory_snapshot(step: int) -> Response:
    """Return memory snapshot data for the given step."""

    try:
        data = await asyncio.to_thread(
            SnapshotPersistenceService.load, step, directory=SNAPSHOT_DIR
        )
    except Exception:  # pragma: no cover - invalid or missing snapshot
        resp = JSONResponse({"error": "not_found"}, status_code=404)
        if not hasattr(resp, "status_code"):
            resp.status_code = 404
        return resp
    resp = JSONResponse(data)
    if not hasattr(resp, "status_code"):
        resp.status_code = 200
    return resp


@app.get("/api/agent_actions/explain_why")
async def api_agent_action_explain_why(after_step: int = 0, limit: int = 20) -> Response:
    """Expose explain-why payloads for recent agent actions."""

    def _load_events() -> list[dict[str, Any]]:
        return event_log.fetch_events(after_step=after_step, event_type="agent_action")

    events = await asyncio.to_thread(_load_events)
    if limit > 0:
        events = events[-limit:]

    agent_events: list[dict[str, Any]] = []
    for evt in events:
        explain = evt.get("explain_why")
        if not isinstance(explain, dict):
            explain = {}
        kb_entries = explain.get("knowledge_board_entries", [])
        if not isinstance(kb_entries, list):
            kb_entries = [kb_entries] if kb_entries else []
        agent_events.append(
            {
                "agent_id": str(evt.get("agent_id", "")),
                "step": int(evt.get("step", evt.get("tick", 0) or 0)),
                "action_intent": evt.get("action_intent"),
                "explain_why": {
                    "memories": explain.get("memories", []),
                    "knowledge_board_entries": [str(entry) for entry in kb_entries],
                    "tool_calls": explain.get("tool_calls", []),
                    "rag_summary": explain.get("rag_summary"),
                },
            }
        )

    resp = JSONResponse({"events": agent_events})
    if not hasattr(resp, "status_code"):
        resp.status_code = 200
    return resp


@app.get("/api/flagged_messages")
async def api_flagged_messages(limit: int = 20) -> Response:
    """Return flagged messages with snapshot references."""

    events = await asyncio.to_thread(event_log.fetch_events)
    flagged = [e for e in events if e.get("type") == "flagged_message"]
    flagged = flagged[-limit:]
    messages: list[dict[str, Any]] = []
    for evt in flagged:
        step = int(evt.get("step", evt.get("tick", 0)))
        msg = str(evt.get("message", ""))
        try:
            event_log.store_replay_slice(step, step, directory=SNAPSHOT_DIR)
        except Exception:  # pragma: no cover - best effort
            pass
        messages.append(
            {
                "step": step,
                "message": msg,
                "snapshot": f"snapshot_{step}.json",
            }
        )

    resp = JSONResponse({"messages": messages})
    if not hasattr(resp, "status_code"):
        resp.status_code = 200
    return resp


@app.get("/api/misbehavior")
async def api_misbehavior(limit: int = 20) -> Response:
    """Return misbehavior events with replay slice paths."""

    events = await asyncio.to_thread(event_log.fetch_events, event_type="misbehavior")
    events = events[-limit:]
    mis_events: list[dict[str, Any]] = []
    for evt in events:
        step = int(evt.get("step", evt.get("tick", 0)))
        agent_id = str(evt.get("agent_id", ""))
        reason = str(evt.get("reason", ""))
        path = evt.get("replay_path")
        if not isinstance(path, str) or not path:
            try:
                slice_path = event_log.store_replay_slice(step, step, directory=SNAPSHOT_DIR)
                path = str(slice_path)
            except Exception:  # pragma: no cover - best effort
                path = ""
        event_data = {"step": step, "agent_id": agent_id, "reason": reason}
        if path:
            event_data["replay_path"] = path
        mis_events.append(event_data)

    resp = JSONResponse({"events": mis_events})
    if not hasattr(resp, "status_code"):
        resp.status_code = 200
    return resp


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


async def handle_control_command(
    cmd: dict[str, Any], ctx: SimulationContext = DEFAULT_CONTEXT
) -> dict[str, Any]:
    """Process dashboard control command through the canonical command service."""
    simulation = ctx.sim_state.get("simulation")
    if simulation is None or getattr(simulation, "command_service", None) is None:
        action = str(cmd.get("command") or "")
        if action == "pause":
            ctx.sim_state["paused"] = True
        elif action == "resume":
            ctx.sim_state["paused"] = False
        elif action == "set_speed":
            try:
                ctx.sim_state["speed"] = float(cmd.get("value", ctx.sim_state.get("speed", 1.0)))
            except (TypeError, ValueError):
                pass
        elif action == "set_breakpoints":
            tags = cmd.get("tags")
            if isinstance(tags, list):
                BREAKPOINT_TAGS.clear()
                BREAKPOINT_TAGS.update(str(t) for t in tags)
        return {**ctx.sim_state, "breakpoints": list(BREAKPOINT_TAGS)}

    from src.interfaces.interaction_schema import InteractionContext

    context = InteractionContext(
        sender_id="dashboard",
        source="dashboard",
        permissions={"admin", "moderator"},
    )
    command = command_from_payload(cmd, context=context)
    result = await simulation.command_dispatcher.dispatch(command, context=context)
    return {
        "status": result.status,
        "message": result.user_message,
        "reason_code": result.reason_code,
        "data": result.data,
        "decision_provenance": result.decision_provenance.model_dump(),
    }


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

        result = await handle_control_command(command, ctx=DEFAULT_CONTEXT)
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
                result = await handle_control_command(cmd, ctx=DEFAULT_CONTEXT)
                await websocket.send_text(json.dumps(result))
        except WebSocketDisconnect:
            pass
        finally:
            await websocket.close()

except AttributeError:  # pragma: no cover - stub app may lack decorators
    pass


def _coerce_tags(value: Any) -> list[str] | None:
    """Normalize tag values into a list for span attributes."""

    if value is None:
        return None
    if isinstance(value, (list, set, tuple)):
        return [str(tag) for tag in value]
    return [str(value)]


async def _publish_with_span(
    bus: Any,
    event: SimulationEvent,
    span_name: str,
    breakpoint_tags: set[str],
) -> None:
    """Publish ``event`` on ``bus`` while recording tracing metadata."""

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("event.type", event.type)

        data = event.data or {}
        step = data.get("step") if isinstance(data, dict) else None
        if step is not None:
            span.set_attribute("event.step", step)

        tags_attr = None
        if isinstance(data, dict):
            tags_attr = _coerce_tags(data.get("tags"))
        if tags_attr is not None:
            span.set_attribute("event.tags", tags_attr)

        span.set_attribute(
            "event.breakpoint_tags", sorted(breakpoint_tags) if breakpoint_tags else []
        )

        await bus.publish(event)


async def emit_event(event: SimulationEvent) -> None:
    """Emit a simulation event and check for breakpoints."""
    bus = get_event_bus()
    tags = set(event.data.get("tags", [])) if event.data else set()
    breakpoint_tags = tags & BREAKPOINT_TAGS

    await _publish_with_span(bus, event, "dashboard.emit_event", breakpoint_tags)

    if breakpoint_tags:
        DEFAULT_CONTEXT.sim_state["paused"] = True
        await _publish_with_span(
            bus,
            SimulationEvent(
                type="breakpoint_hit",
                data={
                    "tags": list(breakpoint_tags),
                    "step": event.data.get("step") if event.data else None,
                },
            ),
            "dashboard.emit_event.breakpoint",
            breakpoint_tags,
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
    "DEFAULT_CONTEXT",
    "WIDGET_REGISTRY",
    "EventSourceResponse",
    "LawProposal",
    "LawsResponse",
    "Proposal",
    "ProposalRecord",
    "ProposalsResponse",
    "SimulationEvent",
    "StakeRequest",
    "VoteRecord",
    "VoteRequest",
    "VotesResponse",
    "api_active_offices",
    "api_agent_stats",
    "api_auctions",
    "api_current_rules",
    "api_flagged_messages",
    "api_get_gov",
    "api_get_laws",
    "api_get_proposals",
    "api_get_votes",
    "api_map",
    "api_memory",
    "api_memory_snapshot",
    "api_memory_snapshots",
    "api_misbehavior",
    "api_pending_votes",
    "api_propose_law",
    "api_recent_proposals",
    "api_sanctions",
    "api_stake_ip",
    "api_token_balances",
    "api_vote",
    "app",
    "board_payload_to_embed",
    "emit_event",
    "emit_map_action_event",
    "emit_map_change_event",
    "enqueue_message",
    "get_event_bus",
    "get_event_queue",
    "get_quests_api",
    "message_sse_queue",
    "register_widget",
    "stream_agents",
    "stream_map",
]
