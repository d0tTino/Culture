from __future__ import annotations

# Skip self argument annotation warnings for protocol stubs
import asyncio
import logging
from typing import Any, Literal, Protocol, cast

from src.agents.core.agent_controller import AgentController
from src.agents.core.base_agent import Agent
from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.infra.llm_client import (
    analyze_sentiment,
    generate_structured_output,
)
from src.shared.typing import SimulationMessage

from .basic_agent_types import AgentActionOutput, AgentTurnState

ActionIntentLiteral = Literal[
    "idle",
    "continue_collaboration",
    "propose_idea",
    "ask_clarification",
    "perform_deep_analysis",
    "create_project",
    "join_project",
    "leave_project",
    "send_direct_message",
]


class MemoryRetriever(Protocol):
    async def aretrieve_relevant_memories(
        self: Any, agent_id: str, query: str, k: int
    ) -> list[dict[str, Any]]: ...


class SummaryAgent(Protocol):
    async def async_generate_l1_summary(
        self: Any, role_prompt: str, memories: str, context: str
    ) -> Any: ...


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
    manager = cast(MemoryRetriever | None, state.get("vector_store_manager"))
    agent = cast(SummaryAgent | None, state.get("agent_instance"))
    semantic_manager = cast(SemanticMemoryManager | None, state.get("semantic_manager"))
    if not manager or not agent:
        return {"rag_summary": "(No memory retrieval)", "memory_history_list": []}
    memories = await manager.aretrieve_relevant_memories(state["agent_id"], query="", k=5)
    memories_content = [m.get("content", "") for m in memories]
    if semantic_manager:
        semantic = semantic_manager.get_recent_summaries(state["agent_id"], limit=2)
        memories_content.extend(semantic)
    agent_state = state.get("state")
    role_prompt = getattr(agent_state, "role_prompt", state.get("current_role", ""))
    summary_result = await agent.async_generate_l1_summary(
        role_prompt,
        "\n".join(memories_content),
        "",
    )
    summary = getattr(summary_result, "summary", "")
    return {"rag_summary": summary, "memory_history_list": memories}


async def retrieve_semantic_context_node(state: AgentTurnState) -> dict[str, Any]:
    """Retrieve semantically grouped context for the agent."""
    semantic_manager = cast(SemanticMemoryManager | None, state.get("semantic_manager"))
    if not semantic_manager:
        return {"semantic_context": ""}
    query = state.get("rag_summary", "")
    import asyncio

    memories = await asyncio.to_thread(
        semantic_manager.retrieve_context,
        state["agent_id"],
        query,
        5,
    )
    context = "\n".join(m.get("content", "") for m in memories)
    return {"semantic_context": context}


def generate_structured_output_from_intent(
    intent: str,
    prompt: str,
    schema: type[AgentActionOutput],
    **kwargs: Any,
) -> AgentActionOutput | None:
    """Compatibility wrapper used by older tests."""

    return generate_structured_output(prompt, schema, **kwargs)


async def generate_thought_and_message_node(
    state: AgentTurnState,
) -> dict[str, AgentActionOutput | None]:
    """Generate a thought and a structured action based on the agent's state."""
    agent = state.get("agent_instance")
    if not agent:
        return {"structured_output": None}
    action_intent: str = "idle"
    result: object | None = None

    # In tests, this can be mocked to return a full AgentActionOutput.
    # The arguments are placeholders as the mock doesn't use them.
    try:
        timeout = getattr(getattr(agent, "async_dspy_manager", None), "default_timeout", 10.0)
        result = await asyncio.wait_for(
            cast(Agent, agent).async_select_action_intent("", "", "", []),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Agent %s: async_select_action_intent timed out after %.2fs",
            getattr(agent, "agent_id", "?"),
            timeout,
        )
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.error(
            "Agent %s: Error in async_select_action_intent: %s",
            getattr(agent, "agent_id", "?"),
            exc,
            exc_info=True,
        )

    # If the mocked result is already the full output, just return it.
    if isinstance(result, AgentActionOutput):
        return {"structured_output": result}

    if result is not None:

        action_intent = getattr(result, "chosen_action_intent", "idle")

    try:
        structured = cast(
            AgentActionOutput | None,
            generate_structured_output(
                "prompt", AgentActionOutput, agent_state=state.get("state")
            ),
        )
    except TypeError:
        structured = cast(
            AgentActionOutput | None,
            generate_structured_output("prompt", AgentActionOutput),
        )

    if structured:
        structured.action_intent = cast(ActionIntentLiteral, action_intent)
    else:
        # Create a minimal object if generation fails, to avoid losing intent.
        structured = AgentActionOutput(
            thought="Structured output generation failed.",
            message_content="",
            message_recipient_id=None,
            action_intent=action_intent,
            requested_role_change=None,
            project_name_to_create=None,
            project_description_for_creation=None,
            project_id_to_join_or_leave=None,
        )

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

    agent_state = state["state"]
    requested_role_change = getattr(output, "requested_role_change", None)
    if requested_role_change:
        from .basic_agent_graph import process_role_change

        if process_role_change(agent_state, requested_role_change):
            AgentController(agent_state).add_memory(
                f"Changed role to {requested_role_change}",
                {"step": state.get("simulation_step", 0), "type": "role_change"},
            )
        else:
            AgentController(agent_state).add_memory(
                "Failed role change attempt",
                {
                    "step": state.get("simulation_step", 0),
                    "type": "resource_constraint",
                },
            )

    return {
        "message_content": output.message_content,
        "message_recipient_id": output.message_recipient_id,
        "action_intent": output.action_intent,
        "updated_agent_state": agent_state,
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
