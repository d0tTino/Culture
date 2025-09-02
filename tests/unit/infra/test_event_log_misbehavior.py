import sys
from types import SimpleNamespace

import pytest

from src.infra.event_log import fetch_events, log_event, log_misbehavior


class MockSpan:
    def __init__(self, name):
        self.name = name
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MockTracer:
    def __init__(self):
        self.spans: list[MockSpan] = []

    def start_as_current_span(self, name):
        span = MockSpan(name)
        self.spans.append(span)
        return span


@pytest.mark.unit
def test_log_misbehavior_and_fetch(monkeypatch, tmp_path):
    tracer = MockTracer()
    monkeypatch.setattr("src.infra.event_log.tracer", tracer)
    monkeypatch.setenv("ENABLE_REDPANDA", "0")
    log_file = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_file))
    monkeypatch.setattr("src.infra.event_log._last_hash", None, raising=False)
    monkeypatch.setattr("src.infra.event_log._seed", None, raising=False)
    monkeypatch.setattr("src.infra.event_log._header_written", False, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "src.infra.checkpoint",
        SimpleNamespace(capture_rng_state=lambda: {}),
    )

    log_event({"type": "normal", "step": 1})
    mis = log_misbehavior({"step": 2, "detail": "bad"})
    assert mis["type"] == "misbehavior"
    assert {"seed", "prev_hash", "trace_hash"}.issubset(mis)

    span = next(s for s in tracer.spans if s.name == "event_log.misbehavior")
    assert span.attributes["event.type"] == "misbehavior"
    assert span.attributes["step"] == 2
    assert {"seed", "prev_hash", "trace_hash"}.issubset(span.attributes)

    # Misbehavior events are filtered out by default
    events = fetch_events(after_step=0, path=log_file)
    assert len(events) == 1
    assert events[0]["type"] == "normal"

    # They can be explicitly included
    all_events = fetch_events(after_step=0, include_misbehavior=True, path=log_file)
    assert [ev["type"] for ev in all_events] == ["normal", "misbehavior"]

    # Or filtered directly
    mis_events = fetch_events(after_step=0, event_type="misbehavior", path=log_file)
    assert len(mis_events) == 1
    assert mis_events[0]["type"] == "misbehavior"
