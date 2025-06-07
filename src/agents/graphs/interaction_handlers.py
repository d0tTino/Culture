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
from src.shared.memory_store import MemoryStore

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


def handle_create_project_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_join_project_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_leave_project_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_send_direct_message_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


# --- Simple vertical slice handlers ---

_RELATIONSHIP_INCREMENT = 0.1
_RELATIONSHIP_REPAIR_BOOST = 1.5
_UNLOCK_THRESHOLD = 0.5
_UNLOCKED_CAPABILITY = "collaborate"


def handle_propose_idea(
    agent: AgentAttributes, memory_store: MemoryStore, knowledge_board: MemoryStore
) -> None:
    """Post an idea to the knowledge board and update agent resources."""
    idea_text = f"Idea from {agent.id}"
    metadata = {"author": agent.id}
    docs = [idea_text]
    metas = [metadata]
    knowledge_board.add_documents(docs, metas)
    memory_store.add_documents(docs, metas)
    agent.ip -= IP_COST_TO_POST_IDEA
    agent.du += config.DU_AWARD_FOR_PROPOSAL


def handle_retrieve_and_update(agent: AgentAttributes, memory_store: MemoryStore) -> None:
    """Retrieve latest idea and adjust relationship score."""
    results = memory_store.query("All inter-agent communications", top_k=1)
    if not results:
        return
    meta = results[0]["metadata"]
    author = meta.get("author")
    if not author:
        return
    previous = agent.relationships.get(author, 0.0)
    delta = _RELATIONSHIP_INCREMENT
    if previous < 0.0 and delta > 0.0:
        delta *= _RELATIONSHIP_REPAIR_BOOST

    new_score = max(-1.0, min(1.0, previous + delta))
    agent.relationships[author] = new_score

    # Update momentum
    momentum_prev = agent.relationship_momentum.get(author, 0.0)
    agent.relationship_momentum[author] = momentum_prev + delta

    # Unlock capability based on threshold
    if new_score >= _UNLOCK_THRESHOLD:
        agent.unlocked_capabilities.add(_UNLOCKED_CAPABILITY)
