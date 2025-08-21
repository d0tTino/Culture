import pytest
import tiktoken

from src.agents.memory.memory_service import MemoryService
from src.agents.memory.multi_layer_retriever import MultiLayerRetriever

pytestmark = pytest.mark.unit


class DummyVectorStore:
    async def aretrieve_relevant_memories(self, agent_id: str, query: str, k: int) -> list[dict[str, str]]:
        return [
            {"content": "episodic one", "relevance_score": 0.9},
            {"content": "episodic two", "relevance_score": 0.8},
        ]


class DummySemantic:
    def retrieve_context_with_scores(self, agent_id: str, query: str, k: int) -> list[dict[str, str]]:
        return [
            {"content": "semantic one", "relevance_score": 0.85},
            {"content": "semantic two", "relevance_score": 0.7},
        ]


@pytest.mark.asyncio
async def test_retriever_respects_token_budget() -> None:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    retriever = MultiLayerRetriever(DummyVectorStore(), DummySemantic(), tokenizer)
    results = await retriever.retrieve("agent", "q", k=10, token_budget=6)
    assert [r["content"] for r in results] == ["episodic one", "semantic one"]


@pytest.mark.asyncio
async def test_context_pipeline_respects_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MemoryService()

    async def fake_retrieve(agent_id: str, query: str, k: int) -> list[dict[str, str]]:
        return [{"content": "one"}, {"content": "two"}, {"content": "three four"}]

    def fake_semantic(agent_id: str, limit: int) -> list[str]:
        return ["alpha", "beta"]

    monkeypatch.setattr(service, "retrieve_episodic_and_update_semantic", fake_retrieve)
    monkeypatch.setattr(service, "get_recent_semantic_summaries", fake_semantic)

    episodic, semantic = await service.get_context_pipeline(
        "agent", token_budget=3, semantic_limit=2
    )
    assert [m["content"] for m in episodic] == ["one", "two"]
    assert semantic == ["alpha"]
