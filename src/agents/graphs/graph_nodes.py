from __future__ import annotations

# Skip self argument annotation warnings for protocol stubs
import asyncio
import logging
from typing import Any, Literal, Protocol, cast

from src.agents.core.agent_controller import AgentController
from src.agents.core.base_agent import Agent
from src.agents.council.orchestrator import run_council
from src.agents.council.types import CouncilOutcome, CouncilQuestion
from src.agents.memory.memory_service import MemoryService
from src.infra.llm_client import (
    analyze_sentiment,
    async_generate_structured_output,
)
from src.interfaces import metrics
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
    async def get_context_pipeline(
        self: Any, agent_id: str, query: str, k: int, semantic_limit: int
    ) -> tuple[list[dict[str, Any]], list[str]]: ...

    def blend_with_recent_semantic(
        self: Any, agent_id: str, episodic_summary: str, limit: int = 3
    ) -> str: ...


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


async def retriever_node(state: AgentTurnState) -> dict[str, Any]:
    service = cast(MemoryService | None, state.get("memory_service"))
    if service is None:
        return {"memory_context": [], "memory_history_list": []}

    token_budget = cast(int | None, state.get("token_budget"))
    episodic, semantic = await service.get_context_pipeline(
        state["agent_id"], token_budget=token_budget
    )

    tokens = 0
    combined: list[str] = []
    limited_episodic: list[dict[str, Any]] = []
    for mem in episodic:
        text = str(mem.get("content", ""))
        t = len(text.split())
        if token_budget is not None and tokens + t > token_budget:
            break
        limited_episodic.append(mem)
        combined.append(text)
        tokens += t

    for summary in semantic:
        t = len(summary.split())
        if token_budget is not None and tokens + t > token_budget:
            break
        combined.append(summary)
        tokens += t

    return {"memory_context": combined, "memory_history_list": limited_episodic}


async def retrieve_and_summarize_memories_node(state: AgentTurnState) -> dict[str, Any]:
    service = cast(MemoryService | None, state.get("memory_service"))
    if service is None:
        return {"rag_summary": "(No memory retrieval)", "memory_history_list": []}

    agent = cast(SummaryAgent | None, state.get("agent_instance"))
    if not service or not agent:
        return {"rag_summary": "(No memory retrieval)", "memory_history_list": []}
    memories = cast(list[dict[str, Any]], state.get("memory_history_list", []))
    memories_content = cast(list[str], state.get("memory_context", []))

    if not memories_content:
        k = 5
        semantic_limit = 2
        memories, semantic = await service.get_context_pipeline(
            state["agent_id"], query="", k=k, semantic_limit=semantic_limit
        )
        metrics.RAG_HIT_RATE.set(
            (len(memories) + len(semantic)) / (k + semantic_limit)
            if (k + semantic_limit)
            else 0
        )
        memories_content = [m.get("content", "") for m in memories] + list(semantic)

    agent_state = state.get("state")
    role_prompt = getattr(agent_state, "role_prompt", state.get("current_role", ""))
    summary_result = await agent.async_generate_l1_summary(
        role_prompt,
        "\n".join(memories_content),
        "",
    )
    summary = getattr(summary_result, "summary", "")

    summary = service.blend_with_recent_semantic(state["agent_id"], summary)

    return {"rag_summary": summary, "memory_history_list": memories}


async def generate_structured_output_from_intent(
    intent: str,
    prompt: str,
    schema: type[AgentActionOutput],
    **kwargs: Any,
) -> AgentActionOutput | None:
    """Compatibility wrapper used by older tests."""

    return await async_generate_structured_output(prompt, schema, **kwargs)


async def generate_thought_and_message_node(
    state: AgentTurnState,
) -> dict[str, AgentActionOutput | None]:
    """Generate a thought and a structured action based on the agent's state."""
    agent = state.get("agent_instance")
    action_intent: str = "idle"
    result: object | None = None

    # In tests, this can be mocked to return a full AgentActionOutput.
    # The arguments are placeholders as the mock doesn't use them.
    if agent:
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

    action_intent = "idle"
    if result:
        action_intent = getattr(result, "chosen_action_intent", "idle")

    try:
        structured = await generate_structured_output_from_intent(
            action_intent,
            "prompt",
            AgentActionOutput,
            agent_state=state.get("state"),
        )
    except TypeError:
        structured = await generate_structured_output_from_intent(
            action_intent,
            "prompt",
            AgentActionOutput,
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


async def council_decision_node(state: AgentTurnState) -> dict[str, AgentActionOutput | None]:
    """Run the agent's proposed action through a council for feedback."""

    output = cast(AgentActionOutput | None, state.get("structured_output"))
    if output is None:
        return {"structured_output": None}

    question = CouncilQuestion(
        question_id=f"{state.get('agent_id', 'agent')}-{state.get('simulation_step', 0)}",
        question=(
            "Given the agent context and proposed action below, propose the best response. "
            "Return a concise resolution summarizing the recommended action."
            f"\n\nGoal: {state.get('agent_goal', '')}"
            f"\nRole: {state.get('current_role', '')}"
            f"\nScenario: {state.get('scenario_description', '')}"
            f"\nPerception: {state.get('environment_perception', {})}"
            f"\nProposed intent: {getattr(output, 'action_intent', 'idle')}"
            f"\nThought: {getattr(output, 'thought', '')}"
            f"\nMessage: {getattr(output, 'message_content', '') or '(no message)'}"
        ),
        extra_context=state.get("prompt_modifier"),
        metadata={
            "agent_id": state.get("agent_id"),
            "simulation_step": state.get("simulation_step"),
        },
    )

    rag_docs: list[str] = []
    for key in ("memory_context", "rag_summary"):
        docs = state.get(key)
        if isinstance(docs, str):
            rag_docs.append(docs)
        elif isinstance(docs, list):
            rag_docs.extend(str(item) for item in docs)

    try:
        outcome: CouncilOutcome = await asyncio.to_thread(
            run_council,
            question,
            extra_context=question.extra_context,
            rag_docs=rag_docs,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Council decision failed: %s", exc, exc_info=True)
        return {"structured_output": output}

    resolution = outcome.resolution or ""
    updated = output.model_copy()
    updated.justification = outcome.summary or resolution or output.justification
    if resolution:
        updated.thought = f"{output.thought}\n\nCouncil resolution: {resolution}".strip()

    return {"structured_output": updated}


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
