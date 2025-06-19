import time

import pytest

from src.infra import config
from src.interfaces import metrics
from src.shared.decorator_utils import monitor_llm_call
from src.sim.knowledge_board import KnowledgeBoard


@pytest.mark.unit
def test_llm_metrics_update() -> None:
    before = metrics.LLM_CALLS_TOTAL._value.get()

    @monitor_llm_call()
    def dummy() -> None:
        time.sleep(0.01)

    dummy()
    after = metrics.LLM_CALLS_TOTAL._value.get()
    assert after == before + 1
    assert metrics.LLM_LATENCY_MS._value.get() > 0


@pytest.mark.unit
def test_kb_size_metric() -> None:
    kb = KnowledgeBoard()
    start = metrics.KNOWLEDGE_BOARD_SIZE._value.get()
    kb.add_entry("entry", "agent", 1)
    assert metrics.KNOWLEDGE_BOARD_SIZE._value.get() == start + 1


@pytest.mark.unit
def test_kb_size_metric_pruning(monkeypatch: pytest.MonkeyPatch) -> None:
    kb = KnowledgeBoard()

    # Add several entries with the default max limit (100)
    kb.add_entry("e1", "A", 1)
    kb.add_entry("e2", "A", 2)
    kb.add_entry("e3", "A", 3)
    before = metrics.KNOWLEDGE_BOARD_SIZE._value.get()

    # Now lower the max size and add another entry to trigger pruning
    monkeypatch.setattr(config, "MAX_KB_ENTRIES", 2)
    kb.add_entry("e4", "A", 4)

    after = metrics.KNOWLEDGE_BOARD_SIZE._value.get()
    assert after == before - 1
    assert len(kb.entries) == 2
