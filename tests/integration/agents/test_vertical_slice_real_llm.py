import asyncio

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("weaviate")
pytest.importorskip("langgraph")

from src.app import create_simulation
from src.infra import llm_client


@pytest.mark.integration
@pytest.mark.skipif(
    not llm_client.is_ollama_available(),
    reason="Ollama service is not available",
)
def test_vertical_slice_real_llm() -> None:
    sim = create_simulation(num_agents=3, steps=2, use_vector_store=True)
    asyncio.run(sim.async_run(sim.steps_to_run))
    assert len(sim.knowledge_board.get_full_entries()) >= 1
