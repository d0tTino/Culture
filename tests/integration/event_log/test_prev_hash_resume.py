import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.infra import event_log


class DummyProducer:
    def __init__(self, sink: list[bytes]) -> None:
        self.sink = sink

    def produce(self, topic: str, payload: bytes) -> None:
        self.sink.append(payload)

    def poll(self, timeout: float) -> None:
        pass


@pytest.mark.integration
@pytest.mark.parametrize("enable_redpanda", [False, True], ids=["file_only", "redpanda_enabled"])
def test_prev_hash_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enable_redpanda: bool) -> None:
    log_file = tmp_path / "events.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(log_file))
    monkeypatch.setattr(event_log, "_last_hash", None, raising=False)
    monkeypatch.setattr(event_log, "_seed", None, raising=False)
    monkeypatch.setattr(event_log, "_seed_cache", {}, raising=False)
    monkeypatch.setattr(event_log, "_header_written", set(), raising=False)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "src.infra.checkpoint",
        SimpleNamespace(capture_rng_state=lambda: {}),
    )

    produced: list[bytes] = []
    if enable_redpanda:
        monkeypatch.setenv("ENABLE_REDPANDA", "1")
        producer = DummyProducer(produced)
        monkeypatch.setattr(event_log, "_get_producer", lambda: producer)
    else:
        monkeypatch.setenv("ENABLE_REDPANDA", "0")

    first = event_log.log_event({"type": "test", "step": 1, "payload": "initial"})

    # Simulate a fresh process where module globals are unset.
    event_log._last_hash = None
    event_log._seed = None
    event_log._seed_cache = {}
    event_log._header_written = set()
    event_log._producer = None

    resumed = event_log.log_event({"type": "test", "step": 2, "payload": "resumed"})
    assert resumed["prev_hash"] == first["trace_hash"]

    if enable_redpanda:
        decoded = [json.loads(item.decode("utf-8")) for item in produced]
        assert decoded[-1]["prev_hash"] == first["trace_hash"]

    # Replaying from disk should include the resumed event thanks to the chained hash.
    monkeypatch.setenv("ENABLE_REDPANDA", "0")
    events = event_log.fetch_events(after_step=0, path=log_file)
    assert [ev["step"] for ev in events] == [1, 2]
    assert events[1]["prev_hash"] == events[0]["trace_hash"]

    with log_file.open("r", encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    # First line is the header; ensure on-disk events are chained.
    assert raw[2]["prev_hash"] == raw[1]["trace_hash"] == first["trace_hash"]
