from contextlib import contextmanager
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


@pytest.mark.unit
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

    if sim._event_listener_task:
        sim._event_listener_task.cancel()
    if sim._event_task:
        sim._event_task.cancel()

    await sim._run_agent_turn(0)

    span = tracer.spans[0]
    assert span.name == "simulation.agent_action"
    assert span.attributes["llm.tokens.prompt"] == 0
    assert span.attributes["llm.tokens.completion"] == 0
    assert "simulation.latency_ms" in span.attributes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluation_hook_tracing(monkeypatch):
    tracer = MockTracer()
    monkeypatch.setattr("src.sim.simulation.tracer", tracer)

    @contextmanager
    def noop_trace_agent_action(*args, **kwargs):
        yield

    monkeypatch.setattr("src.sim.simulation.trace_agent_action", noop_trace_agent_action)
    monkeypatch.setattr(
        "src.sim.simulation.log_event",
        lambda data: {**data, "trace_hash": "hash"},
    )
    monkeypatch.setattr("src.sim.simulation.emit_event", AsyncMock())

    rm = SimpleNamespace(
        cap_tick=lambda **kwargs: None, set_du_budget=lambda *args, **kwargs: None
    )
    monkeypatch.setattr("src.sim.simulation.get_resource_manager", lambda: rm)

    class DummyKernel:
        def __init__(self):
            self._empty = True

        def empty(self):
            return self._empty

        def schedule_immediate_nowait(self, func, **kwargs):
            self._empty = False

        async def step(self, max_turns):
            self._empty = True
            return []

    sim = Simulation(
        [DummyAgent()], memory_service=SimpleNamespace(vector_store=None, semantic_manager=None)
    )
    sim.resource_manager = rm
    sim.event_kernel = DummyKernel()
    sim.vector = SimpleNamespace(increment=lambda agent_id: None)
    sim.start_event_listener = AsyncMock()

    def hook(_sim, _events):
        return {"test_metric": 1}

    sim.evaluation_hooks = [hook]

    await sim.run_step()

    if sim._event_listener_task:
        sim._event_listener_task.cancel()
    if sim._event_task:
        sim._event_task.cancel()

    span = tracer.spans[0]
    assert span.name == "simulation.evaluation_hook"
    assert span.attributes["hook.name"] == "hook"
    assert span.attributes["metric.test_metric"] == 1
    assert span.attributes["simulation.step"] == sim.current_step
