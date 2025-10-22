import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.infra import event_log


@pytest.mark.unit
def test_redpanda_fallback_when_kafka_missing(monkeypatch, tmp_path):
    log_file = tmp_path / "event_log.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_file))
    monkeypatch.setenv("ENABLE_REDPANDA", "1")

    monkeypatch.setitem(
        sys.modules,
        "src.infra.checkpoint",
        SimpleNamespace(capture_rng_state=lambda: {}),
    )

    monkeypatch.setattr(event_log, "KafkaProducer", Any)
    monkeypatch.setattr(event_log, "KafkaConsumer", Any)
    monkeypatch.setattr(event_log, "_KAFKA_IMPORT_ERROR", RuntimeError("missing"))
    monkeypatch.setattr(event_log, "_KAFKA_WARNING_EMITTED", False)
    monkeypatch.setattr(event_log, "_producer", None)
    monkeypatch.setattr(event_log, "_header_written", set())
    monkeypatch.setattr(event_log, "_seed_cache", {})
    monkeypatch.setattr(event_log, "_seed", None)
    monkeypatch.setattr(event_log, "_last_hash", None)

    event = {"type": "test", "step": 1}
    logged_event = event_log.log_event(event)

    assert logged_event["type"] == "test"
    assert "trace_hash" in logged_event

    fetched = event_log.fetch_events(after_step=0, path=log_file, include_misbehavior=True)

    assert fetched
    assert fetched[0]["trace_hash"] == logged_event["trace_hash"]
