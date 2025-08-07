import pytest

from src.agents.core.agent_state import AgentState
from src.agents.graphs.graph_nodes import retrieve_and_summarize_memories_node
from src.interfaces import metrics

pytestmark = pytest.mark.unit


class MockMemoryService:
    async def get_context_pipeline(self, agent_id, query="", k=5, semantic_limit=2):
        return ([{"content": "e1"}], ["s1"])

    def blend_with_recent_semantic(self, agent_id, summary, limit=3):
        return f"blended:{summary}"


class MockSummaryAgent:
    async def async_generate_l1_summary(self, role_prompt, memories, context):
        class R:
            summary = "episodic + semantic"

        return R()


@pytest.mark.asyncio
async def test_retrieve_and_summarize_blends_and_tracks_hit_rate():
    metrics.RAG_HIT_RATE.set(0)
    state = AgentState(agent_id="a1", name="A1")
    turn_state = {
        "agent_id": "a1",
        "state": state,
        "memory_service": MockMemoryService(),
        "agent_instance": MockSummaryAgent(),
    }
    result = await retrieve_and_summarize_memories_node(turn_state)
    assert result["rag_summary"] == "blended:episodic + semantic"
    assert result["memory_history_list"] == [{"content": "e1"}]
    assert metrics.get_rag_hit_rate() == pytest.approx(2 / 7)


class MockMemoryServiceFull:
    async def get_context_pipeline(self, agent_id, query="", k=5, semantic_limit=2):
        episodic = [{"content": f"e{i}"} for i in range(5)]
        semantic = [f"s{i}" for i in range(2)]
        return episodic, semantic

    def blend_with_recent_semantic(self, agent_id, summary, limit=3):
        return f"blended:{summary}"


@pytest.mark.asyncio
async def test_retrieve_and_summarize_full_hit_rate():
    metrics.RAG_HIT_RATE.set(0)
    state = AgentState(agent_id="a1", name="A1")
    turn_state = {
        "agent_id": "a1",
        "state": state,
        "memory_service": MockMemoryServiceFull(),
        "agent_instance": MockSummaryAgent(),
    }
    result = await retrieve_and_summarize_memories_node(turn_state)
    assert result["rag_summary"] == "blended:episodic + semantic"
    assert len(result["memory_history_list"]) == 5
    assert metrics.get_rag_hit_rate() == pytest.approx(1.0)
