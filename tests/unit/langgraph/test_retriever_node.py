import pytest
import tiktoken

from src.agents.core.agent_state import AgentState


class DummyMemoryService:
    async def retrieve_relevant_memories(self, agent_id: str, query: str, k: int):
        return [
            {"content": "こんにちは世界"},
            {"content": "hello world"},
        ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retriever_node_respects_token_cap_non_ascii() -> None:
    service = DummyMemoryService()
    agent_state = AgentState(agent_id="a1", name="A1")
    agent_state.memory_retriever_top_k = 5
    agent_state.memory_retriever_token_cap = 5
    node = agent_state.get_retriever_node(service)
    result = await node({"agent_id": "a1", "query": ""})
    assert len(result["memories"]) == 1
    combined = " ".join(m["content"] for m in result["memories"])
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(combined)) <= agent_state.memory_retriever_token_cap
