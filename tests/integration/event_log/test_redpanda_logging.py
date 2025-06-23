import json
from typing import Any

import pytest

from src.infra import event_log


class DummyProducer:
    def __init__(self, store: list[bytes]):
        self.store = store

    def produce(self, topic: str, payload: bytes) -> None:
        self.store.append(payload)

    def poll(self, timeout: float) -> None:  # pragma: no cover - no-op
        return None


class DummyConsumer:
    def __init__(self, _conf: dict[str, Any], store: list[bytes]):
        self.store = store
        self.index = 0

    def subscribe(self, _topics: list[str]) -> None:  # pragma: no cover - no-op
        pass

    def poll(self, _timeout: float):
        if self.index >= len(self.store):
            return None
        payload = self.store[self.index]
        self.index += 1

        class Message:
            def __init__(self, value: bytes) -> None:
                self._value = value

            def error(self) -> None:
                return None

            def value(self) -> bytes:
                return self._value

        return Message(payload)

    def close(self) -> None:  # pragma: no cover - no-op
        pass


@pytest.mark.integration
def test_log_and_fetch_events(monkeypatch):
    messages: list[bytes] = []
    monkeypatch.setenv("ENABLE_REDPANDA", "1")
    monkeypatch.setattr(event_log, "KafkaProducer", lambda conf: DummyProducer(messages))
    monkeypatch.setattr(event_log, "KafkaConsumer", lambda conf: DummyConsumer(conf, messages))
    monkeypatch.setattr(event_log, "_producer", None, raising=False)

    events_in = [
        {"step": 1, "msg": "first", "trace_hash": "aaa"},
        {"step": 2, "msg": "second", "trace_hash": "bbb"},
        {"step": 3, "msg": "third", "trace_hash": "ccc"},
    ]

    for ev in events_in:
        event_log.log_event(ev)

    events_out = event_log.fetch_events(after_step=0)
    assert [json.loads(m) for m in messages] == events_in
    assert events_out == events_in
    assert all("trace_hash" in ev for ev in events_out)

    after_first = event_log.fetch_events(after_step=1)
    assert after_first == events_in[1:]
    assert all("trace_hash" in ev for ev in after_first)
