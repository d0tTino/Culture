import httpx
import pytest
from httpx import ASGITransport

pytest.importorskip("fastapi")

from src import http_app
from src.interfaces import dashboard_backend as db


class RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, object] = {}

    def __enter__(self) -> "RecordingSpan":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    def start_as_current_span(self, name: str) -> RecordingSpan:
        span = RecordingSpan(name)
        self.spans.append(span)
        return span


class FailingManager:
    def get_semantic_summaries(self, agent_id: str, limit: int = 3) -> list[str]:
        raise RuntimeError("boom")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_summaries_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(db.SIM_STATE, "semantic_manager", FailingManager())
    transport = ASGITransport(app=http_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/agents/agent-1/semantic_summaries")
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"] == "summary retrieval failed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_emit_event_tracing_records_attributes_for_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = RecordingTracer()
    monkeypatch.setattr(db, "tracer", tracer)
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "paused", False)

    await db.emit_event(db.SimulationEvent(type="evaluation"))
    await db.emit_event(db.SimulationEvent(type="evaluation", data={"step": 3}))

    assert [span.name for span in tracer.spans] == [
        "dashboard.emit_event",
        "dashboard.emit_event",
    ]
    empty_span, step_span = tracer.spans

    assert empty_span.attributes["event.type"] == "evaluation"
    assert empty_span.attributes["event.breakpoint_tags"] == []
    assert "event.step" not in empty_span.attributes

    assert step_span.attributes["event.type"] == "evaluation"
    assert step_span.attributes["event.step"] == 3
    assert step_span.attributes["event.breakpoint_tags"] == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_emit_event_tracing_records_breakpoint_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = RecordingTracer()
    monkeypatch.setattr(db, "tracer", tracer)
    monkeypatch.setitem(db.DEFAULT_CONTEXT.sim_state, "paused", False)

    await db.emit_event(
        db.SimulationEvent(type="evaluation", data={"step": 5, "tags": ["nsfw", "other"]})
    )

    assert [span.name for span in tracer.spans] == [
        "dashboard.emit_event",
        "dashboard.emit_event.breakpoint",
    ]

    original_span, breakpoint_span = tracer.spans
    assert original_span.attributes["event.type"] == "evaluation"
    assert original_span.attributes["event.step"] == 5
    assert original_span.attributes["event.breakpoint_tags"] == ["nsfw"]

    assert breakpoint_span.attributes["event.type"] == "breakpoint_hit"
    assert breakpoint_span.attributes["event.step"] == 5
    assert breakpoint_span.attributes["event.breakpoint_tags"] == ["nsfw"]

    assert db.DEFAULT_CONTEXT.sim_state["paused"] is True
    db.DEFAULT_CONTEXT.sim_state["paused"] = False
