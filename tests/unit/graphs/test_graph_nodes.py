from types import SimpleNamespace
from typing import cast

import pytest

from src.agents.graphs.graph_nodes import (
    _format_knowledge_board,
    _format_other_agents,
    analyze_perception_sentiment_node,
    finalize_message_agent_node,
    generate_thought_and_message_node,
    prepare_relationship_prompt_node,
    retrieve_and_summarize_memories_node,
)


@pytest.mark.unit
def test_analyze_perception_sentiment_node(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_sentiment(text: str) -> float:
        calls.append(text)
        return 1.0 if text == "good" else -1.0

    monkeypatch.setattr("src.agents.graphs.graph_nodes.analyze_sentiment", fake_sentiment)

    state = {
        "agent_id": "a",
        "perceived_messages": [
            {"sender_id": "b", "content": "good"},
            {"sender_id": "c", "content": "bad"},
        ],
    }
    result = analyze_perception_sentiment_node(state)
    assert result == {"turn_sentiment_score": 0}
    assert calls == ["good", "bad"]


@pytest.mark.unit
def test_prepare_relationship_prompt_node() -> None:
    agent_state = SimpleNamespace(relationships={"b": 0.5, "c": -0.2})
    result = prepare_relationship_prompt_node(cast(dict[str, object], {"state": agent_state}))
    assert "b: 0.5" in result["prompt_modifier"]
    assert "c: -0.2" in result["prompt_modifier"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_and_summarize_memories_node_no_manager() -> None:
    state = {"agent_id": "a"}
    out = await retrieve_and_summarize_memories_node(cast(object, state))
    assert out["rag_summary"] == "(No memory retrieval)"
    assert out["memory_history_list"] == []


class DummyService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int]] = []

    async def get_context_pipeline(
        self, agent_id: str, query: str = "", k: int = 5, semantic_limit: int = 2
    ) -> tuple[list[dict[str, str]], list[str]]:
        self.calls.append((agent_id, query, k, semantic_limit))
        return [{"content": "m1"}, {"content": "m2"}], ["sem1"]

    def blend_with_recent_semantic(
        self, agent_id: str, episodic_summary: str, limit: int = 3
    ) -> str:
        return f"{episodic_summary}\nsem1"


class DummyAgent:
    async def async_generate_l1_summary(
        self, role: str, memories: str, context: str
    ) -> SimpleNamespace:
        return SimpleNamespace(summary="SUM")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_and_summarize_memories_node_with_manager() -> None:
    service = DummyService()
    state = {
        "agent_id": "a",
        "memory_service": service,
        "agent_instance": DummyAgent(),
        "current_role": "r",
    }
    out = await retrieve_and_summarize_memories_node(cast(object, state))
    assert service.calls == [("a", "", 5, 2)]
    assert out["rag_summary"] == "SUM\nsem1"
    assert out["memory_history_list"] == [{"content": "m1"}, {"content": "m2"}]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_generate_thought_and_message_node(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyOutput(SimpleNamespace):
        pass

    dummy = DummyOutput(
        thought="T",
        message_content="M",
        message_recipient_id=None,
        action_intent="continue_collaboration",
    )

    monkeypatch.setattr(
        "src.agents.graphs.graph_nodes.generate_structured_output_from_intent",
        lambda intent, prompt, schema: dummy,
    )

    out = await generate_thought_and_message_node(cast(dict[str, object], {}))
    assert out == {"structured_output": dummy}


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.require_ollama
async def test_finalize_message_agent_node_variants() -> None:
    pytest.skip("skip in CI")
    agent_state = SimpleNamespace()
    out = await finalize_message_agent_node(cast(dict[str, object], {"state": agent_state}))
    assert out["message_content"] is None
    assert out["action_intent"] == "idle"

    dummy = SimpleNamespace(
        message_content="hi",
        message_recipient_id="b",
        action_intent="propose",
    )
    out2 = await finalize_message_agent_node(cast(dict[str, object], {"state": agent_state, "structured_output": dummy}))
    assert out2["message_content"] == "hi"
    assert out2["is_targeted"] is True


@pytest.mark.unit
def test_helper_formatters() -> None:
    info = [{"agent_id": "b"}]
    relationships = {"b": 0.2}
    assert _format_other_agents(info, relationships) == "b: 0.2"
    assert _format_other_agents([], relationships) == "None"
    assert _format_knowledge_board(["a", "b"]) == "a | b"
    assert _format_knowledge_board([]) == "(Board empty)"
