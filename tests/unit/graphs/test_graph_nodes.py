from types import SimpleNamespace

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

    def fake_sentiment(text: str) -> str:
        calls.append(text)
        return "positive" if text == "good" else "negative"

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
    result = prepare_relationship_prompt_node({"state": agent_state})
    assert "b: 0.5" in result["prompt_modifier"]
    assert "c: -0.2" in result["prompt_modifier"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_and_summarize_memories_node_no_manager() -> None:
    state = {"agent_id": "a"}
    out = await retrieve_and_summarize_memories_node(state)
    assert out == {"rag_summary": "(No memory retrieval)"}


class DummyManager:
    async def aretrieve_relevant_memories(
        self, agent_id: str, query: str = "", k: int = 5
    ) -> list[dict[str, str]]:
        return [{"content": "m1"}, {"content": "m2"}]


class DummyAgent:
    async def async_generate_l1_summary(
        self, role: str, memories: str, context: str
    ) -> SimpleNamespace:
        return SimpleNamespace(summary="SUM")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieve_and_summarize_memories_node_with_manager() -> None:
    state = {
        "agent_id": "a",
        "vector_store_manager": DummyManager(),
        "agent_instance": DummyAgent(),
        "current_role": "r",
    }
    out = await retrieve_and_summarize_memories_node(state)
    assert out == {"rag_summary": "SUM"}


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
        "src.agents.graphs.graph_nodes.generate_structured_output", lambda prompt, schema: dummy
    )

    out = await generate_thought_and_message_node({})
    assert out == {"structured_output": dummy}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_finalize_message_agent_node_variants() -> None:
    agent_state = SimpleNamespace()
    out = await finalize_message_agent_node({"state": agent_state})
    assert out["message_content"] is None
    assert out["action_intent"] == "idle"

    dummy = SimpleNamespace(
        message_content="hi",
        message_recipient_id="b",
        action_intent="propose",
    )
    out2 = await finalize_message_agent_node({"state": agent_state, "structured_output": dummy})
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
