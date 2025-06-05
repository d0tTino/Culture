from pathlib import Path

import pytest

from src.agents.core.agent_attributes import AgentAttributes
from src.agents.graphs.interaction_handlers import handle_retrieve_and_update
from src.shared.memory_store import ChromaMemoryStore


@pytest.mark.integration
def test_agent_b_retrieve_and_update(tmp_path: Path) -> None:
    memory = ChromaMemoryStore(persist_directory=str(tmp_path))
    memory.add_documents(["Idea from A"], [{"author": "A", "timestamp": 0}])

    state_b = AgentAttributes(id="B", mood=0.0, goals=[], resources={}, relationships={"A": 0.0})

    handle_retrieve_and_update(state_b, memory)

    assert state_b.relationships["A"] > 0.0
