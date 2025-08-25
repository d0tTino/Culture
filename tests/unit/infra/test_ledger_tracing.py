from pathlib import Path

import pytest

from src.infra.ledger import Ledger


class MockSpan:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        return False


class MockTracer:
    def __init__(self) -> None:
        self.spans: list[MockSpan] = []

    def start_as_current_span(self, name: str):
        span = MockSpan(name)
        self.spans.append(span)
        return span


pytestmark = pytest.mark.unit


def test_ledger_emits_spans(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tracer = MockTracer()
    monkeypatch.setattr("src.infra.ledger.tracer", tracer)

    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger.log_change("a1", 5.0, 2.0, "init")
    ledger.get_balance("a1")

    span_names = [span.name for span in tracer.spans]
    assert "Ledger.log_change" in span_names
    assert "Ledger.get_balance" in span_names
