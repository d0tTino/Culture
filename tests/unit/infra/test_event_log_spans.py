import sys
from types import SimpleNamespace

import pytest

from src.infra.event_log import fetch_events, log_event, stream_events


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
@pytest.mark.parametrize("use_redpanda", [False, True])
def test_fetch_events_tracing(monkeypatch, tmp_path, use_redpanda):
    tracer = MockTracer()
    monkeypatch.setattr("src.infra.event_log.tracer", tracer)
    if use_redpanda:
        monkeypatch.setenv("ENABLE_REDPANDA", "1")

        class DummyConsumer:
            def __init__(self, conf):
                pass

            def subscribe(self, topics):
                pass

            def poll(self, timeout):
                return None

            def close(self):
                return None

        monkeypatch.setattr("src.infra.event_log.KafkaConsumer", DummyConsumer)
        fetch_events(after_step=1, end_step=5)
    else:
        monkeypatch.setenv("ENABLE_REDPANDA", "0")
        log_file = tmp_path / "event_log.jsonl"
        log_file.write_text("")
        fetch_events(after_step=1, end_step=5, path=log_file)

    span = tracer.spans[0]
    assert span.name == "event_log.fetch_events"
    assert span.attributes["tick.start"] == 1
    assert span.attributes["tick.end"] == 5
    expected_source = "redpanda" if use_redpanda else "file"
    assert span.attributes["event.source"] == expected_source


@pytest.mark.unit
def test_log_event_tracing(monkeypatch, tmp_path):
    tracer = MockTracer()
    monkeypatch.setattr("src.infra.event_log.tracer", tracer)
    monkeypatch.setenv("ENABLE_REDPANDA", "0")
    log_file = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_file))
    monkeypatch.setitem(
        sys.modules,
        "src.infra.checkpoint",
        SimpleNamespace(capture_rng_state=lambda: {}),
    )

    event = {"type": "test", "step": 7}
    log_event(event)

    span = tracer.spans[0]
    assert span.name == "event_log.log_event"
    assert span.attributes["event.type"] == "test"
    assert span.attributes["step"] == 7
    assert span.attributes["log.file_path"] == str(log_file)


@pytest.mark.unit
@pytest.mark.parametrize("use_redpanda", [False, True])
def test_stream_events_tracing(monkeypatch, tmp_path, use_redpanda):
    tracer = MockTracer()
    monkeypatch.setattr("src.infra.event_log.tracer", tracer)
    if use_redpanda:
        monkeypatch.setenv("ENABLE_REDPANDA", "1")

        class DummyConsumer:
            def __init__(self, conf):
                pass

            def subscribe(self, topics):
                pass

            def poll(self, timeout):
                return None

            def close(self):
                return None

        monkeypatch.setattr("src.infra.event_log.KafkaConsumer", DummyConsumer)
        list(stream_events(after_step=2, end_step=6, timeout=0.01))
    else:
        monkeypatch.setenv("ENABLE_REDPANDA", "0")
        log_file = tmp_path / "event_log.jsonl"
        log_file.write_text("")
        list(stream_events(after_step=2, end_step=6, path=log_file))

    span = tracer.spans[0]
    assert span.name == "event_log.stream_events"
    assert span.attributes["tick.start"] == 2
    assert span.attributes["tick.end"] == 6
    expected_source = "redpanda" if use_redpanda else "file"
    assert span.attributes["event.source"] == expected_source
