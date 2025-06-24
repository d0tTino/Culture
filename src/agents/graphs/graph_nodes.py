from __future__ import annotations

import logging
from typing import Any, cast

from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.infra.llm_client import analyze_sentiment, generate_structured_output
from src.shared.typing import SimulationMessage

from .basic_agent_types import AgentActionOutput, AgentTurnState

logger = logging.getLogger(__name__)


def analyze_perception_sentiment_node(state: AgentTurnState) -> dict[str, Any]:
    agent_id = state["agent_id"]
    perceived_messages = cast(list[SimulationMessage], state.get("perceived_messages", []))
    total = 0
    for msg in perceived_messages:
        if msg.get("sender_id") == agent_id:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            try:
                sentiment = analyze_sentiment(content, agent_state=state.get("state"))
            except TypeError:
                sentiment = analyze_sentiment(content)
            if isinstance(sentiment, str):
                mapping = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                sentiment = mapping.get(sentiment.lower(), 0.0)
            if sentiment is not None:
                if sentiment > 0:
                    total += 1
                elif sentiment < 0:
                    total -= 1
    return {"turn_sentiment_score": total}


def prepare_relationship_prompt_node(state: AgentTurnState) -> dict[str, str]:
    relationships = state["state"].relationships
    if not relationships:
        return {"prompt_modifier": "You have neutral relationships."}
    lines = [f"- {aid}: {score:.1f}" for aid, score in relationships.items()]
    return {"prompt_modifier": "Relationships:\n" + "\n".join(lines)}


async def retrieve_and_summarize_memories_node(state: AgentTurnState) -> dict[str, Any]:
    manager = state.get("vector_store_manager")
    agent = state.get("agent_instance")
    semantic_manager: SemanticMemoryManager | None = state.get("semantic_manager")
    if not manager or not agent:
        return {"rag_summary": "(No memory retrieval)", "memory_history_list": []}
    memories = await manager.aretrieve_relevant_memories(  # type: ignore[attr-defined]
        state["agent_id"], query="", k=5
    )
    memories_content = [m.get("content", "") for m in memories]
    if semantic_manager:
        semantic = semantic_manager.get_recent_summaries(state["agent_id"], limit=2)
        memories_content.extend(semantic)
    agent_state = state.get("state")
    role_prompt = getattr(agent_state, "role_prompt", state.get("current_role", ""))
    summary_result = await agent.async_generate_l1_summary(  # type: ignore[attr-defined]
        role_prompt,
        "\n".join(memories_content),
        "",
    )
    summary = getattr(summary_result, "summary", "")
    return {"rag_summary": summary, "memory_history_list": memories}


async def generate_thought_and_message_node(
    state: AgentTurnState,
) -> dict[str, AgentActionOutput | None]:
    agent = state.get("agent_instance")
    thought = "Thinking..."
    action_intent = "idle"
    if agent and hasattr(agent, "async_select_action_intent"):
        result = await agent.async_select_action_intent("", "", "", [])
        if result:
            action_intent = getattr(result, "chosen_action_intent", "idle")
    try:
        structured = generate_structured_output(
            "prompt", AgentActionOutput, agent_state=state.get("state")
        )
    except TypeError:
        structured = generate_structured_output("prompt", AgentActionOutput)
    return {"structured_output": cast(AgentActionOutput | None, structured)}


async def finalize_message_agent_node(state: AgentTurnState) -> dict[str, Any]:
    output = state.get("structured_output")
    if not output:
        return {
            "message_content": None,
            "message_recipient_id": None,
            "action_intent": "idle",
            "updated_agent_state": state["state"],
            "memory_history_list": state.get("memory_history_list", []),
        }
    return {
        "message_content": output.message_content,
        "message_recipient_id": output.message_recipient_id,
        "action_intent": output.action_intent,
        "updated_agent_state": state["state"],
        "is_targeted": output.message_recipient_id is not None,
        "memory_history_list": state.get("memory_history_list", []),
    }


# Helper formatters


def _format_other_agents(
    other_agents_info: list[dict[str, Any]], relationships: dict[str, float]
) -> str:
    if not other_agents_info:
        return "None"
    lines = []
    for info in other_agents_info:
        other_id = info.get("agent_id", "?")
        score = relationships.get(other_id, 0.0)
        lines.append(f"{other_id}: {score:.1f}")
    return " | ".join(lines)


def _format_knowledge_board(board_entries: list[str]) -> str:
    return " | ".join(board_entries) if board_entries else "(Board empty)"
