from pathlib import Path

import pytest

from src.agents.core.agent_attributes import AgentAttributes
from src.agents.graphs.interaction_handlers import handle_propose_idea, handle_retrieve_and_update
from src.shared.memory_store import ChromaMemoryStore


@pytest.mark.integration
def test_full_a_then_b(tmp_path: Path) -> None:
    memory = ChromaMemoryStore(persist_directory=str(tmp_path))
    state_a = AgentAttributes(
        id="A", mood=0.0, goals=["innovate"], resources={}, relationships={"B": 0.0}
    )
    state_b = AgentAttributes(id="B", mood=0.0, goals=[], resources={}, relationships={"A": 0.0})

    handle_propose_idea(state_a, memory, memory)
    assert state_a.relationships["B"] == 0.0

    handle_retrieve_and_update(state_b, memory)
    assert state_b.relationships["A"] > 0.0
