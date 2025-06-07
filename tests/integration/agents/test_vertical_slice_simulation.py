import asyncio

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("weaviate")
pytest.importorskip("langgraph")

from src.agents.dspy_programs.intent_selector import _StubLM
from src.app import create_simulation
from src.infra.dspy_ollama_integration import dspy


@pytest.mark.integration
def test_vertical_slice_simulation() -> None:
    dspy.settings.configure(lm=_StubLM())
    sim = create_simulation(num_agents=3, steps=2, use_vector_store=True)
    asyncio.run(sim.async_run(sim.steps_to_run))
    assert len(sim.knowledge_board.get_full_entries()) >= 1
    relationships_updated = any(
        any(score != 0.0 for score in agent.state.relationships.values())
        for agent in sim.agents
    )
    assert relationships_updated
