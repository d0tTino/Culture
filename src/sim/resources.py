from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.infra.ledger import ledger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .simulation import Simulation

logger = logging.getLogger(__name__)


def mute_agent(sim: Simulation, agent_id: str) -> dict[str, object]:
    """Mute ``agent_id`` and return the moderation event payload."""

    sim.muted_agents.add(agent_id)
    return {
        "type": "moderation",
        "action": "mute",
        "agent_id": agent_id,
        "step": sim.current_step,
    }


async def unmute_agent(sim: Simulation, agent_id: str) -> None:
    """Unmute ``agent_id`` and broadcast the action."""
    sim.muted_agents.discard(agent_id)
    event = {
def unmute_agent(sim: Simulation, agent_id: str) -> dict[str, object]:
    """Unmute ``agent_id`` and return the moderation event payload."""

    sim.muted_agents.discard(agent_id)
    return {
        "type": "moderation",
        "action": "unmute",
        "agent_id": agent_id,
        "step": sim.current_step,
    }
    await sim.event_kernel.emit_environment_event(event)


async def reset_memory(sim: Simulation, agent_id: str) -> None:
    """Reset the memory of ``agent_id`` and broadcast the action."""


def reset_memory(sim: Simulation, agent_id: str) -> dict[str, object] | None:
    """Reset the memory of ``agent_id`` and return the moderation event payload."""

    try:
        sim.memory_service.reset_agent(agent_id)
    except Exception:  # pragma: no cover - defensive
        logger.error("Failed to reset memory for %s", agent_id, exc_info=True)
        return None
    return {
        "type": "moderation",
        "action": "reset_memory",
        "agent_id": agent_id,
        "step": sim.current_step,
    }


def apply_penalty(
    sim: Simulation,
    agent_id: str,
    ip: float = 0.0,
    du: float = 0.0,
) -> dict[str, object] | None:
    """Apply an IP/DU penalty to ``agent_id`` and return the moderation event payload."""

    penalty_ip = abs(float(ip))
    penalty_du = abs(float(du))

    for agent in sim.agents:
        if agent.agent_id == agent_id:
            state = getattr(agent, "state", None)
            if state is not None:
                if hasattr(state, "ip"):
                    state.ip = max(float(getattr(state, "ip", 0.0)) - penalty_ip, 0.0)
                if hasattr(state, "du"):
                    state.du = max(float(getattr(state, "du", 0.0)) - penalty_du, 0.0)
            break

    try:
        ledger.log_change(agent_id, -penalty_ip, -penalty_du, "moderation_penalty")
    except Exception:  # pragma: no cover - defensive
        logger.error("Failed to apply penalty to %s", agent_id, exc_info=True)
        return None

    return {
        "type": "moderation",
        "action": "penalty",
        "agent_id": agent_id,
        "ip": float(ip),
        "du": float(du),
        "step": sim.current_step,
    }


__all__ = ["apply_penalty", "mute_agent", "reset_memory", "unmute_agent"]
