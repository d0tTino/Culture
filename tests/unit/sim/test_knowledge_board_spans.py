from __future__ import annotations

from typing import Any

import pytest

from src.sim.knowledge_board import KnowledgeBoard


class MockSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def __enter__(self) -> MockSpan:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        return False


class MockTracer:
    def __init__(self) -> None:
        self.spans: list[MockSpan] = []

    def start_as_current_span(self, name: str) -> MockSpan:
        span = MockSpan(name)
        self.spans.append(span)
        return span


pytestmark = pytest.mark.unit


def test_add_entry_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = MockTracer()
    monkeypatch.setattr("src.shared.telemetry.tracer", tracer)

    kb = KnowledgeBoard()
    kb.add_entry("entry", agent_id="agent", step=1)

    span = tracer.spans[0]
    assert span.name == "agent.knowledge_board.add_entry"
    assert span.attributes["agent.id"] == "agent"
    assert span.attributes["step"] == 1


def test_get_state_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = MockTracer()
    monkeypatch.setattr("src.shared.telemetry.tracer", tracer)

    kb = KnowledgeBoard()
    kb.get_state(2)

    span = tracer.spans[0]
    assert span.name == "agent.knowledge_board.get_state"
    assert span.attributes["max_entries"] == 2


def test_get_recent_entries_for_prompt_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = MockTracer()
    monkeypatch.setattr("src.shared.telemetry.tracer", tracer)

    kb = KnowledgeBoard()
    kb.entries.append(
        {
            "entry_id": "id",
            "step": 1,
            "agent_id": "agent",
            "content_full": "entry",
            "content_display": "entry",
            "content_summary": "entry",
        }
    )

    kb.get_recent_entries_for_prompt(1)

    span = tracer.spans[0]
    assert span.name == "agent.knowledge_board.get_recent_entries_for_prompt"
    assert span.attributes["max_entries"] == 1
