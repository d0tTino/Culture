import asyncio
import sys
import types

import pytest
from pytest import MonkeyPatch

from src.agents.dspy_programs.intent_selector import _StubLM
from src.infra.dspy_ollama_integration import dspy
from src.shared import llm_mocks


@pytest.mark.integration
def test_walking_vertical_slice(monkeypatch: MonkeyPatch) -> None:
    """Run a small simulation with mocked LLM responses."""
    llm_mocks.patch_ollama_functions(monkeypatch)

    # Patch vector store and dashboard modules to avoid heavy dependencies
    vector_store_stub = types.ModuleType("src.agents.memory.vector_store")
    weaviate_store_stub = types.ModuleType("src.agents.memory.weaviate_vector_store_manager")
    dashboard_stub = types.ModuleType("src.interfaces.dashboard_backend")

    class _StubVectorStoreManager:
        def __init__(self, *args: object, **kwargs: object) -> None:
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

    monkeypatch.setitem(sys.modules, "src.agents.memory.vector_store", vector_store_stub)
    monkeypatch.setitem(
        sys.modules,
        "src.agents.memory.weaviate_vector_store_manager",
        weaviate_store_stub,
    )
    monkeypatch.setitem(sys.modules, "src.interfaces.dashboard_backend", dashboard_stub)

    from src.agents.core.agent_state import AgentState

    AgentState.update_collective_metrics = lambda self, ip, du: None

    # Stub langgraph and graph compilation
    langgraph_stub = types.ModuleType("langgraph.graph")
    langgraph_stub.StateGraph = object
    langgraph_stub.END = "END"
    monkeypatch.setitem(sys.modules, "langgraph.graph", langgraph_stub)

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

    bag_stub = types.ModuleType("src.agents.graphs.basic_agent_graph")
    bag_stub.compile_agent_graph = lambda: stub_executor
    monkeypatch.setitem(sys.modules, "src.agents.graphs.basic_agent_graph", bag_stub)

    from src.app import create_simulation

    dspy.settings.configure(lm=_StubLM())
    sim = create_simulation(
        num_agents=3,
        steps=3,
        scenario="Vertical slice demonstration",
        use_vector_store=False,
    )
    asyncio.run(sim.async_run(sim.steps_to_run))

    assert len(sim.knowledge_board.get_full_entries()) >= 1
    relationships_updated = any(
        any(score != 0.0 for score in agent.state.relationships.values()) for agent in sim.agents
    )
    assert relationships_updated
