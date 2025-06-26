from __future__ import annotations

from typing import Any

from src.infra import config

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - optional dependency
    END = "END"
    StateGraph: Any = Any  # type: ignore[no-redef]

from .basic_agent_graph import _maybe_consolidate_memories
from .basic_agent_types import AgentTurnState
from .graph_nodes import (
    analyze_perception_sentiment_node,
    finalize_message_agent_node,
    generate_thought_and_message_node,
    prepare_relationship_prompt_node,
    retrieve_and_summarize_memories_node,
    retrieve_semantic_context_node,
)
from .interaction_handlers import (
    handle_ask_clarification_node,
    handle_continue_collaboration_node,
    handle_create_project_node,  # - imported for future use
    handle_deep_analysis_node,
    handle_idle_node,
    handle_join_project_node,  # - imported for future use
    handle_leave_project_node,  # - imported for future use
    handle_propose_idea_node,
    handle_send_direct_message_node,  # - imported for future use
)


def route_action_intent(state: AgentTurnState) -> str:
    output = state.get("structured_output")
    intent = getattr(output, "action_intent", "idle") if output else "idle"
    return intent


def build_graph() -> Any:
    graph_builder = StateGraph(AgentTurnState)
    graph_builder.add_node("analyze_perception_sentiment", analyze_perception_sentiment_node)
    graph_builder.add_node("prepare_relationship_prompt", prepare_relationship_prompt_node)
    graph_builder.add_node("retrieve_and_summarize_memories", retrieve_and_summarize_memories_node)
    graph_builder.add_node("retrieve_semantic_context", retrieve_semantic_context_node)
    graph_builder.add_node("generate_thought_and_message", generate_thought_and_message_node)

    graph_builder.add_node("route_action_intent", route_action_intent)

    graph_builder.add_node("handle_propose_idea", handle_propose_idea_node)
    graph_builder.add_node("handle_ask_clarification", handle_ask_clarification_node)
    graph_builder.add_node("handle_continue_collaboration", handle_continue_collaboration_node)
    graph_builder.add_node("handle_idle", handle_idle_node)
    graph_builder.add_node("handle_deep_analysis", handle_deep_analysis_node)
    graph_builder.add_node("handle_create_project", handle_create_project_node)
    graph_builder.add_node("handle_join_project", handle_join_project_node)
    graph_builder.add_node("handle_leave_project", handle_leave_project_node)
    graph_builder.add_node("handle_send_direct_message", handle_send_direct_message_node)

    graph_builder.add_node("finalize_message_agent", finalize_message_agent_node)
    consolidation_interval = config.get_config_value_with_override(
        "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS", 0
    )
    if int(consolidation_interval) > 0:
        graph_builder.add_node("maybe_consolidate_memories", _maybe_consolidate_memories)

    graph_builder.set_entry_point("analyze_perception_sentiment")
    graph_builder.add_edge("analyze_perception_sentiment", "prepare_relationship_prompt")
    graph_builder.add_edge("prepare_relationship_prompt", "retrieve_and_summarize_memories")
    graph_builder.add_edge("retrieve_and_summarize_memories", "retrieve_semantic_context")
    graph_builder.add_edge("retrieve_semantic_context", "generate_thought_and_message")
    graph_builder.add_edge("generate_thought_and_message", "route_action_intent")

    graph_builder.add_conditional_edges(
        "route_action_intent",
        route_action_intent,
        {
            "propose_idea": "handle_propose_idea",
            "ask_clarification": "handle_ask_clarification",
            "continue_collaboration": "handle_continue_collaboration",
            "idle": "handle_idle",
            "perform_deep_analysis": "handle_deep_analysis",
            "create_project": "handle_create_project",
            "join_project": "handle_join_project",
            "leave_project": "handle_leave_project",
            "send_direct_message": "handle_send_direct_message",
        },
    )

    for node in [
        "handle_propose_idea",
        "handle_ask_clarification",
        "handle_continue_collaboration",
        "handle_idle",
        "handle_deep_analysis",
        "handle_create_project",
        "handle_join_project",
        "handle_leave_project",
        "handle_send_direct_message",
    ]:
        graph_builder.add_edge(node, "finalize_message_agent")

    graph_builder.add_edge("finalize_message_agent", END)
    consolidation_interval = config.get_config_value_with_override(
        "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS", 0
    )
    if int(consolidation_interval) > 0:
        graph_builder.add_edge("finalize_message_agent", "maybe_consolidate_memories")
        graph_builder.add_edge("maybe_consolidate_memories", END)
    return graph_builder.compile()
