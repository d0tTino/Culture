from pathlib import Path

import pytest
from pytest import MonkeyPatch

from src.agents.core.agent_attributes import AgentAttributes
from src.agents.core.agent_controller import AgentController
from src.agents.dspy_programs.intent_selector import _StubLM
from src.agents.graphs.interaction_handlers import (
    handle_propose_idea,
    handle_retrieve_and_update,
)
from src.infra.dspy_ollama_integration import dspy
from src.shared.memory_store import ChromaMemoryStore


@pytest.mark.integration
def test_intent_pipeline(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    class DummyPredict:
        def __init__(self: "DummyPredict", *args: object, **kwargs: object) -> None:
            pass

        def __call__(self: "DummyPredict", *args: object, **kwargs: object) -> object:
            return type("Result", (), {"intent": "PROPOSE_IDEA"})()

    monkeypatch.setattr(dspy, "Predict", DummyPredict, raising=False)

    controller = AgentController(lm=_StubLM())
    state_a = AgentAttributes(id="A", mood=0.0, goals=["innovate"], resources={}, relationships={})
    memory = ChromaMemoryStore(persist_directory=str(tmp_path))
    knowledge_board = memory

    intent = controller.select_intent(state_a)
    assert intent == "PROPOSE_IDEA"

    if intent == "PROPOSE_IDEA":
        handle_propose_idea(state_a, memory, knowledge_board)

    results = knowledge_board.query("All inter-agent communications", top_k=1)
    assert len(results) == 1
    assert results[0]["metadata"].get("author") == "A"

    state_b = AgentAttributes(id="B", mood=0.0, goals=[], resources={}, relationships={"A": -0.2})
    handle_retrieve_and_update(state_b, memory)

    assert state_b.relationships["A"] > -0.2
    assert state_b.relationship_momentum["A"] > 0.0
