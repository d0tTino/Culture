from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from src.sim.resource_manager import get_resource_manager
from src.sim.simulation import Simulation


class _InMemoryExporter(SpanExporter):
    """Simple in-memory span exporter for assertions."""

    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans):  # type: ignore[override]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover - no action needed
        return None


def _setup_tracer() -> _InMemoryExporter:
    exporter = _InMemoryExporter()
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def _llm_call(state: SimpleNamespace) -> None:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("llm.request"):
        get_resource_manager().ensure_du_budget(state.agent_id, 1.0)


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.state = SimpleNamespace(
            agent_id=agent_id,
            ip=0.0,
            du=0.0,
            is_alive=True,
            messages_sent_count=0,
            last_message_step=0,
            short_term_memory=[],
        )

    def get_id(self) -> str:
        return self.agent_id

    def update_state(self, new_state: SimpleNamespace) -> None:  # pragma: no cover - unused
        self.state = new_state

    async def run_turn(self, **_: dict) -> dict:
        _llm_call(self.state)
        return {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_du_budget_llm_request_failure() -> None:
    exporter = _setup_tracer()
    agent = DummyAgent("a1")
    Simulation(agents=[agent])

    with pytest.raises(RuntimeError):
        await agent.run_turn()

    span = next(s for s in exporter.spans if s.name == "llm.request")
    assert span.status.status_code == StatusCode.ERROR
