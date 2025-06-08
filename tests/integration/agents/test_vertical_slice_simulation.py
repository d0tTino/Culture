import asyncio
import sys
import types

import pytest

from src.agents.dspy_programs.intent_selector import _StubLM
from src.infra.dspy_ollama_integration import dspy


@pytest.mark.integration
def test_vertical_slice_simulation() -> None:
    # Patch vector store and dashboard imports to avoid heavy dependencies
    vector_store_stub = types.ModuleType("src.agents.memory.vector_store")
    weaviate_store_stub = types.ModuleType("src.agents.memory.weaviate_vector_store_manager")
    dashboard_stub = types.ModuleType("src.interfaces.dashboard_backend")

    from typing import Any

    class _StubVectorStoreManager:
        """Placeholder stub for vector store managers."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Accept any initialization parameters."""
            pass

    vector_store_stub.ChromaVectorStoreManager = _StubVectorStoreManager
    weaviate_store_stub.WeaviateVectorStoreManager = _StubVectorStoreManager

    from pydantic import BaseModel

    class _AgentMessage(BaseModel):
        agent_id: str = ""
        content: str = ""
        step: int = 0

    dashboard_stub.AgentMessage = _AgentMessage
    dashboard_stub.message_sse_queue = asyncio.Queue()

    sys.modules["src.agents.memory.vector_store"] = vector_store_stub
    sys.modules["src.agents.memory.weaviate_vector_store_manager"] = weaviate_store_stub
    sys.modules["src.interfaces.dashboard_backend"] = dashboard_stub

    from src.agents.core.agent_state import AgentState

    AgentState.update_collective_metrics = lambda self, ip, du: None

    # Stub langgraph and graph compilation to avoid heavy dependencies
    langgraph_stub = types.ModuleType("langgraph.graph")
    langgraph_stub.StateGraph = object  # simple placeholder
    langgraph_stub.END = "END"
    sys.modules["langgraph.graph"] = langgraph_stub

    async def _ainvoke(state: dict) -> dict:
        kb = state.get("knowledge_board")
        if kb is not None:
            kb.add_entry(
                "stub entry",
                agent_id=state["agent_id"],
                step=state.get("simulation_step", 0),
            )
        agent_state = state.get("state")
        if agent_state is not None:
            agent_state.relationships[agent_state.agent_id] = 1.0
        return state

    stub_executor = types.SimpleNamespace(ainvoke=_ainvoke)
    import src.agents.graphs.basic_agent_graph as bag

    bag.compile_agent_graph = lambda: stub_executor

    from src.app import create_simulation

    dspy.settings.configure(lm=_StubLM())
    sim = create_simulation(num_agents=3, steps=2, use_vector_store=False)
    asyncio.run(sim.async_run(sim.steps_to_run))
    assert len(sim.knowledge_board.get_full_entries()) >= 1
    relationships_updated = any(
        any(score != 0.0 for score in agent.state.relationships.values())
        for agent in sim.agents

    )
    assert relationships_updated
