from __future__ import annotations

# mypy: ignore-errors
import logging
from typing import Any

from src.agents.core.agent_attributes import AgentAttributes
from src.infra import config
from src.infra.config import (
    DU_COST_DEEP_ANALYSIS,
    IP_AWARD_FOR_PROPOSAL,
    IP_COST_TO_POST_IDEA,
)
from src.shared.memory_store import ChromaMemoryStore

try:
    from .basic_agent_graph import AgentTurnState
except Exception:  # pragma: no cover - fallback for simplified tests
    AgentTurnState = dict  # type: ignore

logger = logging.getLogger(__name__)


def handle_propose_idea_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state = state["state"]
    agent_state.ip -= IP_COST_TO_POST_IDEA
    agent_state.ip += IP_AWARD_FOR_PROPOSAL
    return dict(state)


def handle_continue_collaboration_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_idle_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_deep_analysis_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state = state["state"]
    agent_state.du -= DU_COST_DEEP_ANALYSIS
    return dict(state)


def handle_ask_clarification_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


# --- Simple vertical slice handlers ---

_RELATIONSHIP_INCREMENT = 0.1


def handle_propose_idea(
    agent: AgentAttributes, memory_store: ChromaMemoryStore, knowledge_board: ChromaMemoryStore
) -> None:
    """Post an idea to the knowledge board and update agent resources."""
    idea_text = f"Idea from {agent.id}"
    metadata = {"author": agent.id}
    knowledge_board.add_documents([idea_text], [metadata])
    agent.ip -= IP_COST_TO_POST_IDEA
    agent.du += config.DU_AWARD_FOR_PROPOSAL


def handle_retrieve_and_update(agent: AgentAttributes, memory_store: ChromaMemoryStore) -> None:
    """Retrieve latest idea and adjust relationship score."""
    results = memory_store.query("All inter-agent communications", top_k=1)
    if not results:
        return
    meta = results[0]["metadata"]
    author = meta.get("author")
    if not author:
        return
    agent.relationships[author] = agent.relationships.get(author, 0.0) + _RELATIONSHIP_INCREMENT
