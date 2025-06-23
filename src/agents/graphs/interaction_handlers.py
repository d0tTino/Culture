from __future__ import annotations

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
    from .basic_agent_types import AgentTurnState
except Exception:  # pragma: no cover - fallback for simplified tests
    # During tests, AgentTurnState may be unavailable; use plain dict
    AgentTurnState = dict  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


def handle_propose_idea_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state = state["state"]
    start_ip = agent_state.ip
    agent_state.ip -= IP_COST_TO_POST_IDEA
    agent_state.ip += IP_AWARD_FOR_PROPOSAL
    try:
        from src.infra.ledger import ledger

        ledger.log_change(
            agent_state.agent_id,
            agent_state.ip - start_ip,
            0.0,
            "propose_idea",
        )
    except Exception:  # pragma: no cover - optional
        logger.debug("Ledger logging failed", exc_info=True)
    return dict(state)


def handle_continue_collaboration_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_idle_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_deep_analysis_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state = state["state"]
    start_du = agent_state.du
    agent_state.du -= DU_COST_DEEP_ANALYSIS
    try:
        from src.infra.ledger import ledger

        ledger.log_change(
            agent_state.agent_id,
            0.0,
            agent_state.du - start_du,
            "deep_analysis",
        )
    except Exception:  # pragma: no cover - optional
        logger.debug("Ledger logging failed", exc_info=True)
    return dict(state)


def handle_ask_clarification_node(state: AgentTurnState) -> dict[str, Any]:
    return dict(state)


def handle_create_project_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state = state["state"]
    structured_output = state.get("structured_output")
    perception = state.get("environment_perception", {})
    simulation = perception.get("simulation")

    if structured_output and simulation is not None and structured_output.project_name_to_create:
        project_name = structured_output.project_name_to_create
        project_description = structured_output.project_description_for_creation

        start_ip = agent_state.ip
        start_du = agent_state.du

        project_id = simulation.create_project(
            project_name, agent_state.agent_id, project_description
        )

        if project_id:
            agent_state.current_project_id = project_id
            agent_state.projects[project_id] = simulation.projects[project_id]
            agent_state.ip -= config.IP_COST_CREATE_PROJECT
            agent_state.du -= config.DU_COST_CREATE_PROJECT
            try:
                from src.infra.ledger import ledger

                ledger.log_change(
                    agent_state.agent_id,
                    agent_state.ip - start_ip,
                    agent_state.du - start_du,
                    "create_project",
                )
            except Exception:  # pragma: no cover - optional
                logger.debug("Ledger logging failed", exc_info=True)

    return dict(state)


def handle_join_project_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state = state["state"]
    structured_output = state.get("structured_output")
    perception = state.get("environment_perception", {})
    simulation = perception.get("simulation")

    if (
        structured_output
        and simulation is not None
        and structured_output.project_id_to_join_or_leave
    ):
        project_id = structured_output.project_id_to_join_or_leave
        start_ip = agent_state.ip
        start_du = agent_state.du

        if simulation.join_project(project_id, agent_state.agent_id):
            project = simulation.projects.get(project_id, {})
            agent_state.current_project_id = project_id
            agent_state.projects[project_id] = project
            agent_state.ip -= config.IP_COST_JOIN_PROJECT
            agent_state.du -= config.DU_COST_JOIN_PROJECT
            try:
                from src.infra.ledger import ledger

                ledger.log_change(
                    agent_state.agent_id,
                    agent_state.ip - start_ip,
                    agent_state.du - start_du,
                    "join_project",
                )
            except Exception:  # pragma: no cover - optional
                logger.debug("Ledger logging failed", exc_info=True)

    return dict(state)


def handle_leave_project_node(state: AgentTurnState) -> dict[str, Any]:
    agent_state = state["state"]
    structured_output = state.get("structured_output")
    perception = state.get("environment_perception", {})
    simulation = perception.get("simulation")

    if (
        structured_output
        and simulation is not None
        and structured_output.project_id_to_join_or_leave
    ):
        project_id = structured_output.project_id_to_join_or_leave
        if simulation.leave_project(project_id, agent_state.agent_id):
            agent_state.current_project_id = None
            agent_state.projects.pop(project_id, None)

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
    start_ip = agent.ip
    start_du = agent.du
    agent.ip -= IP_COST_TO_POST_IDEA
    agent.du += config.DU_AWARD_FOR_PROPOSAL
    try:
        from src.infra.ledger import ledger

        ledger.log_change(
            agent.id,
            agent.ip - start_ip,
            agent.du - start_du,
            "propose_idea",
        )
    except Exception:  # pragma: no cover - optional
        logger.debug("Ledger logging failed", exc_info=True)


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
