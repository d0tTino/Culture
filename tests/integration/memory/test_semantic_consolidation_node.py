import pytest

from src.agents.graphs.basic_agent_graph import _maybe_consolidate_memories
from src.infra import config


class DummyManager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def consolidate_memories(self, agent_id: str) -> None:
        self.calls.append(agent_id)


@pytest.mark.integration
@pytest.mark.memory
def test_consolidation_runs_when_due(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyManager()
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS", 1)
    state = {"agent_id": "agent1", "simulation_step": 1, "semantic_manager": dummy}
    _maybe_consolidate_memories(state)
    assert dummy.calls == ["agent1"]


@pytest.mark.integration
@pytest.mark.memory
def test_consolidation_skipped_when_not_due(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyManager()
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS", 5)
    state = {"agent_id": "agent1", "simulation_step": 1, "semantic_manager": dummy}
    _maybe_consolidate_memories(state)
    assert dummy.calls == []
