from types import SimpleNamespace

import pytest

from src.agents.graphs import agent_graph_builder
from src.infra import config

pytestmark = pytest.mark.unit


def _patch_graph_nodes(monkeypatch: pytest.MonkeyPatch, *, include_council: bool, calls: list[str]) -> None:
    def make_node(name: str, *, set_output: bool = False):  # type: ignore[type-arg]
        async def _node(state: dict) -> dict:
            calls.append(name)
            if set_output:
                state["structured_output"] = SimpleNamespace(action_intent="propose_idea")
                return {"structured_output": state["structured_output"]}
            return {}

        return _node

    for node_name in [
        "analyze_perception_sentiment_node",
        "prepare_relationship_prompt_node",
        "retriever_node",
        "retrieve_and_summarize_memories_node",
        "handle_propose_idea_node",
        "handle_ask_clarification_node",
        "handle_continue_collaboration_node",
        "handle_idle_node",
        "handle_deep_analysis_node",
        "handle_create_project_node",
        "handle_join_project_node",
        "handle_leave_project_node",
        "handle_send_direct_message_node",
        "finalize_message_agent_node",
        "_maybe_consolidate_memories",
    ]:
        monkeypatch.setattr(agent_graph_builder, node_name, make_node(node_name))

    monkeypatch.setattr(
        agent_graph_builder,
        "generate_thought_and_message_node",
        make_node("generate_thought_and_message_node", set_output=True),
    )

    if include_council:
        monkeypatch.setattr(
            agent_graph_builder,
            "council_decision_node",
            make_node("council_decision_node"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_values", "expect_council"),
    [
        ({"USE_COUNCIL_MODE": False, "USE_COUNCIL_FOR_DECISIONS": True}, False),
        ({"USE_COUNCIL_MODE": True, "USE_COUNCIL_FOR_DECISIONS": False}, False),
        ({"USE_COUNCIL_MODE": True, "USE_COUNCIL_FOR_DECISIONS": True}, True),
    ],
)
async def test_graph_council_routing_requires_master_and_scoped_flags(
    monkeypatch: pytest.MonkeyPatch,
    config_values: dict[str, bool],
    expect_council: bool,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(config, "_CONFIG", config_values)
    _patch_graph_nodes(monkeypatch, include_council=expect_council, calls=calls)

    executor = agent_graph_builder.build_graph()
    await executor.ainvoke({})

    assert ("council_decision_node" in calls) is expect_council
    assert "generate_thought_and_message_node" in calls
    # Ensure decision handlers still run after council/no-council routing
    assert any(call.startswith("handle_") for call in calls)
