from __future__ import annotations

# mypy: ignore-errors
import logging
from typing import Any

from src.infra.config import (
    DU_COST_DEEP_ANALYSIS,
    IP_AWARD_FOR_PROPOSAL,
    IP_COST_TO_POST_IDEA,
)

from .basic_agent_graph import AgentTurnState

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
