from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.infra.config import get_config

from .roles import create_role_profile

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from .agent_state import AgentState

logger = logging.getLogger(__name__)


def update_relationship(
    state: AgentState,
    other_agent_id: str,
    sentiment_score: float,
    *,
    is_targeted: bool = False,
) -> None:
    """Update ``state`` relationship score with ``other_agent_id``."""
    current_score = state.relationships.get(other_agent_id, 0.0)
    sentiment_score = float(sentiment_score) if sentiment_score is not None else 0.0
    effective = (
        sentiment_score * state._targeted_message_multiplier if is_targeted else sentiment_score
    )
    if effective > 0:
        lr = state._positive_relationship_learning_rate
    elif effective < 0:
        lr = state._negative_relationship_learning_rate
    else:
        lr = state._neutral_relationship_learning_rate
    change = effective * lr
    new_score = current_score + change
    new_score = max(state._min_relationship_score, min(state._max_relationship_score, new_score))
    state.relationships[other_agent_id] = new_score
    if abs(new_score - current_score) > 0.01:
        state.relationship_history.setdefault(other_agent_id, []).append(
            (state.step_counter, new_score)
        )


def can_change_role(state: AgentState, new_role: str, current_step: int) -> bool:
    if new_role == state.current_role.name:
        logger.debug(
            "AGENT_STATE (%s): Role change to %s denied (already current role).",
            state.agent_id,
            new_role,
        )
        return False
    if state.ip < state._role_change_ip_cost:
        logger.debug(
            "AGENT_STATE (%s): Role change to %s denied (insufficient IP: %.2f < %.2f).",
            state.agent_id,
            new_role,
            state.ip,
            state._role_change_ip_cost,
        )
        return False
    last_change_step = -1
    if len(state.role_history) > 1:
        last_change_step = state.role_history[-1][0]
    if (current_step - last_change_step) < state._role_change_cooldown and last_change_step != -1:
        logger.debug(
            "AGENT_STATE (%s): Role change to %s denied (cooldown). current=%d last=%d cd=%d",
            state.agent_id,
            new_role,
            current_step,
            last_change_step,
            state._role_change_cooldown,
        )
        return False
    role_du_gen = get_config("ROLE_DU_GENERATION")
    if isinstance(role_du_gen, dict) and new_role not in role_du_gen:
        logger.warning(
            "AGENT_STATE (%s): Attempted role change to unrecognized role '%s'.",
            state.agent_id,
            new_role,
        )
        return False
    return True


def change_role(state: AgentState, new_role: str, current_step: int) -> bool:
    if not can_change_role(state, new_role, current_step):
        return False
    start_ip = state.ip
    state.ip -= state._role_change_ip_cost
    logger.info(
        "AGENT_STATE (%s): Role changed from %s to %s. IP cost: %.2f. New IP: %.2f",
        state.agent_id,
        state.current_role.name,
        new_role,
        state._role_change_ip_cost,
        state.ip,
    )
    try:
        from src.infra.ledger import ledger

        ledger.log_change(state.agent_id, state.ip - start_ip, 0.0, "role_change")
    except Exception:  # pragma: no cover - ledger optional
        logger.debug("Ledger logging failed", exc_info=True)
    state.current_role = create_role_profile(new_role)
    state.steps_in_current_role = 0
    state.role_history.append((current_step, new_role))
    return True


__all__ = ["can_change_role", "change_role", "update_relationship"]
