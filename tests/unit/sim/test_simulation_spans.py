from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.sim.simulation import Simulation


class MockSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MockTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        span = MockSpan(name)
        self.spans.append(span)
        return span


class DummyAgent:
    def __init__(self):
        self.agent_id = "agent"
        self.state = SimpleNamespace(
            agent_id="agent",
            ip=1.0,
            du=1.0,
            messages_sent_count=0,
            last_message_step=0,
            short_term_memory=[],
            mood_level=0.0,
            is_alive=True,
        )

    async def run_turn(self, **kwargs):
        return {"action_intent": "idle"}

    def update_state(self, state):
        self.state = state

    def get_id(self):
        return self.agent_id


@pytest.mark.asyncio
async def test_agent_action_tracing(monkeypatch):
    tracer = MockTracer()
    monkeypatch.setattr("src.sim.simulation.tracer", tracer)
    monkeypatch.setattr("src.sim.simulation.evaluate_policy", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "src.sim.simulation.log_event",
        lambda data: {**data, "trace_hash": "hash"},
    )
    monkeypatch.setattr("src.sim.simulation.emit_event", AsyncMock())

    rm = SimpleNamespace(
        cap_tick=lambda **kwargs: None, set_du_budget=lambda *args, **kwargs: None
    )
    monkeypatch.setattr("src.sim.simulation.get_resource_manager", lambda: rm)

    sim = Simulation(
        [DummyAgent()], memory_service=SimpleNamespace(vector_store=None, semantic_manager=None)
    )
    sim.resource_manager = rm

    await sim._run_agent_turn(0)

    span = tracer.spans[0]
    assert span.name == "simulation.agent_action"
    assert span.attributes["llm.tokens.prompt"] == 0
    assert span.attributes["llm.tokens.completion"] == 0
    assert "simulation.latency_ms" in span.attributes
