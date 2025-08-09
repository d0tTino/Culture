import pytest

from src.agents.core.agent_state import AgentState


class DummyMemoryService:
    async def retrieve_relevant_memories(self, agent_id: str, query: str, k: int):
        return [
            {"content": "one two three four"},
            {"content": "five six seven eight"},
            {"content": "nine ten eleven twelve"},
        ]


@pytest.mark.asyncio
async def test_retriever_node_respects_token_cap() -> None:
    service = DummyMemoryService()
    agent_state = AgentState(agent_id="a1", name="A1")
    agent_state.memory_retriever_top_k = 5
    agent_state.memory_retriever_token_cap = 8
    node = agent_state.get_retriever_node(service)
    result = await node({"agent_id": "a1", "query": ""})
    assert len(result["memories"]) == 2
    combined = " ".join(m["content"] for m in result["memories"])
    assert len(combined.split()) <= agent_state.memory_retriever_token_cap
