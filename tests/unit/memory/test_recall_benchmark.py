import time

import pytest

from src.agents.memory.memory_service import MemoryService

pytestmark = pytest.mark.unit


class SimpleVectorStoreManager:
    """Minimal in-memory vector store for testing."""

    def __init__(self) -> None:
        self.memories: list[dict[str, str]] = []

    def add_memory(
        self,
        agent_id: str,
        step: int,
        event_type: str,
        content: str,
        memory_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        self.memories.append({"agent_id": agent_id, "content": content})
        return str(len(self.memories))

    async def aretrieve_relevant_memories(
        self, agent_id: str, query: str, k: int = 5
    ) -> list[dict[str, object]]:
        words = set(query.lower().split())
        results: list[dict[str, object]] = []
        for mem in self.memories:
            if mem["agent_id"] != agent_id:
                continue
            mem_words = set(mem["content"].lower().split())
            score = float(len(words & mem_words))
            results.append({**mem, "relevance_score": score})
        results.sort(key=lambda m: m["relevance_score"], reverse=True)
        return results[:k]


FIXTURE_MEMORIES = [
    "Apples are a type of fruit",
    "Bananas are a type of fruit",
    "Cherries are a type of fruit",
    "Dates are a type of fruit",
    "Elderberries are a type of fruit",
    "Cats are playful animals",
    "Dogs are loyal animals",
    "Elephants are large animals",
    "Foxes are clever animals",
    "Giraffes are tall animals",
]

DATASET = [
    {"query": "fruit", "relevant": set(FIXTURE_MEMORIES[:5])},
    {"query": "animals", "relevant": set(FIXTURE_MEMORIES[5:])},
]


@pytest.mark.asyncio
async def test_retrieve_relevant_memories() -> None:
    """Ensure p@5 meets benchmark and report latency."""
    vector = SimpleVectorStoreManager()
    service = MemoryService(vector_store=vector)

    for i, content in enumerate(FIXTURE_MEMORIES):
        vector.add_memory("agent", i, "test", content)

    precisions: list[float] = []
    latencies: list[float] = []

    for item in DATASET:
        query = item["query"]
        relevant = item["relevant"]

        start = time.perf_counter()
        result = await service.retrieve_relevant_memories("agent", query, k=5)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved = {mem["content"] for mem in result}
        hits = len(retrieved & relevant)
        precision = hits / 5

        precisions.append(precision)
        latencies.append(latency_ms)

        print(f"query={query} p@5={precision:.2f} latency_ms={latency_ms:.2f}")

    avg_precision = sum(precisions) / len(precisions)
    avg_latency = sum(latencies) / len(latencies)
    print(f"avg_p@5={avg_precision:.2f} avg_latency_ms={avg_latency:.2f}")

    assert avg_precision >= 0.7
