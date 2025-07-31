import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sklearn")

from src.agents.memory.memory_service import MemoryService
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager
from src.sim.simulation import Simulation


class _DummyAgent:
    """Minimal agent used for simulation tests."""

    def __init__(self, agent_id: str, service: MemoryService) -> None:
        self.agent_id = agent_id
        self.state = SimpleNamespace(
            ip=0.0,
            du=0.0,
            short_term_memory=[],
            relationships={},
        )
        self.memory_service = service

    async def run_turn(self, simulation_step: int, **_: object) -> dict[str, object]:
        await self.memory_service.get_context_pipeline(self.agent_id, k=2, semantic_limit=1)
        return {"message_content": None, "message_recipient_id": None, "action_intent": "idle"}

    def update_state(self, new_state: object) -> None:
        self.state = new_state  # pragma: no cover - simple assign

    def get_id(self) -> str:  # pragma: no cover - used by Simulation
        return self.agent_id


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.usefixtures("chroma_test_dir")
async def test_multilayer_retriever_in_sim(
    monkeypatch: pytest.MonkeyPatch, chroma_test_dir: Path
) -> None:
    """Run a tiny simulation and verify MultiLayerRetriever behavior."""
    from tests.unit.memory.test_semantic_memory_manager import DummyDriver

    neo4j_stub = types.ModuleType("neo4j")
    neo4j_stub.Driver = DummyDriver
    neo4j_stub.GraphDatabase = types.SimpleNamespace(driver=lambda *a, **k: DummyDriver())
    monkeypatch.setitem(sys.modules, "neo4j", neo4j_stub)
    monkeypatch.setitem(sys.modules, "neo4j.exceptions", types.ModuleType("neo4j.exceptions"))

    vector = ChromaVectorStoreManager(
        persist_directory=chroma_test_dir, embedding_function=lambda t: [[0.0] for _ in t]
    )
    driver = DummyDriver()
    semantic = SemanticMemoryManager(vector, driver)
    service = MemoryService(vector, semantic)

    agent = _DummyAgent("agent_1", service)
    sim = Simulation(
        agents=[agent],
        memory_service=service,
        vector_store_manager=vector,
        semantic_manager=semantic,
    )
    sim.steps_to_run = 1

    # Seed episodic memories
    vector.add_memory("agent_1", 1, "thought", "e1", memory_type="raw")
    vector.add_memory("agent_1", 2, "thought", "e2", memory_type="raw")

    # Patch retrieval functions for deterministic results
    async def fake_episodic(agent_id: str, query: str, k: int) -> list[dict]:
        return [
            {"content": "e1", "relevance_score": 0.5},
            {"content": "e2", "relevance_score": 0.2},
        ]

    def fake_semantic(agent_id: str, query: str, k: int) -> list[dict]:
        return [
            {"content": "s1", "relevance_score": 0.9},
            {"content": "s2", "relevance_score": 0.6},
        ]

    monkeypatch.setattr(vector, "aretrieve_relevant_memories", fake_episodic)
    monkeypatch.setattr(semantic, "retrieve_context_with_scores", fake_semantic)

    summary_calls: list[str] = []

    async def fake_run_job(agent_id: str, memories: list[dict] | None = None) -> None:
        summary_calls.append(agent_id)

    monkeypatch.setattr(semantic, "run_nightly_job", fake_run_job)

    await sim.async_run(sim.steps_to_run)

    results = await service.retriever.retrieve("agent_1", "q", k=4)
    assert [r["content"] for r in results] == ["s1", "s2", "e1", "e2"]
    assert summary_calls == ["agent_1"]
