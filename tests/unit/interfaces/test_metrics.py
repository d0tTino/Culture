import time

import pytest

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
