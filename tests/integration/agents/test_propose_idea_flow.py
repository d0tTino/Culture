from pathlib import Path

import pytest

from src.agents.core.agent_attributes import AgentAttributes
from src.agents.graphs.interaction_handlers import handle_propose_idea
from src.infra import config
from src.shared.memory_store import ChromaMemoryStore


@pytest.mark.integration
def test_agent_a_propose_idea_roundtrip(tmp_path: Path) -> None:
    state_a = AgentAttributes(id="A", mood=0.0, goals=["innovate"], resources={}, relationships={})
    memory = ChromaMemoryStore(persist_directory=str(tmp_path))
    knowledge_board = memory

    handle_propose_idea(state_a, memory, knowledge_board)

    assert state_a.ip == config.INITIAL_INFLUENCE_POINTS - config.IP_COST_TO_POST_IDEA
    assert state_a.du == config.INITIAL_DATA_UNITS + config.DU_AWARD_FOR_PROPOSAL

    results = memory.query("All inter-agent communications", top_k=1)
    assert len(results) == 1
    assert not state_a.unlocked_capabilities
