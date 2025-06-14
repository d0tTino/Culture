from __future__ import annotations

from typing import Any

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - optional dependency
    END = "END"
    StateGraph: Any = Any  # type: ignore[no-redef]

from .basic_agent_types import AgentTurnState
from .graph_nodes import (
    analyze_perception_sentiment_node,
    finalize_message_agent_node,
    generate_thought_and_message_node,
    prepare_relationship_prompt_node,
    retrieve_and_summarize_memories_node,
)
from .interaction_handlers import (
    handle_ask_clarification_node,
    handle_continue_collaboration_node,
    handle_create_project_node,  # noqa: F401 - imported for future use
    handle_deep_analysis_node,
    handle_idle_node,
    handle_join_project_node,  # noqa: F401 - imported for future use
    handle_leave_project_node,  # noqa: F401 - imported for future use
    handle_propose_idea_node,
    handle_send_direct_message_node,  # noqa: F401 - imported for future use
)


def build_graph() -> Any:
    graph_builder = StateGraph(AgentTurnState)
    graph_builder.add_node("analyze_perception_sentiment", analyze_perception_sentiment_node)
    graph_builder.add_node("prepare_relationship_prompt", prepare_relationship_prompt_node)
    graph_builder.add_node("retrieve_and_summarize_memories", retrieve_and_summarize_memories_node)
    graph_builder.add_node("generate_thought_and_message", generate_thought_and_message_node)

    graph_builder.add_node("handle_propose_idea", handle_propose_idea_node)
    graph_builder.add_node("handle_ask_clarification", handle_ask_clarification_node)
    graph_builder.add_node("handle_continue_collaboration", handle_continue_collaboration_node)
    graph_builder.add_node("handle_idle", handle_idle_node)
    graph_builder.add_node("handle_deep_analysis", handle_deep_analysis_node)

    graph_builder.add_node("finalize_message_agent", finalize_message_agent_node)

    graph_builder.set_entry_point("analyze_perception_sentiment")
    graph_builder.add_edge("analyze_perception_sentiment", "prepare_relationship_prompt")
    graph_builder.add_edge("prepare_relationship_prompt", "retrieve_and_summarize_memories")
    graph_builder.add_edge("retrieve_and_summarize_memories", "generate_thought_and_message")
    graph_builder.add_edge("generate_thought_and_message", "handle_idle")
    graph_builder.add_edge("handle_idle", "finalize_message_agent")
    graph_builder.add_edge("finalize_message_agent", END)
    return graph_builder
