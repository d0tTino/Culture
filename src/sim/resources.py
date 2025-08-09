from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.infra.ledger import ledger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .simulation import Simulation

logger = logging.getLogger(__name__)


async def mute_agent(sim: Simulation, agent_id: str) -> None:
    """Mute ``agent_id`` and broadcast the action."""
    sim.muted_agents.add(agent_id)
    event = {
        "type": "moderation",
        "action": "mute",
        "agent_id": agent_id,
        "step": sim.current_step,
    }
    await sim.event_kernel.emit_environment_event(event)


async def reset_memory(sim: Simulation, agent_id: str) -> None:
    """Reset the memory of ``agent_id`` and broadcast the action."""
    try:
        sim.memory_service.reset_agent(agent_id)
    except Exception:  # pragma: no cover - defensive
        logger.error("Failed to reset memory for %s", agent_id, exc_info=True)
        return
    event = {
        "type": "moderation",
        "action": "reset_memory",
        "agent_id": agent_id,
        "step": sim.current_step,
    }
    await sim.event_kernel.emit_environment_event(event)


async def apply_penalty(sim: Simulation, agent_id: str, ip: float = 0.0, du: float = 0.0) -> None:
    """Apply an IP/DU penalty to ``agent_id`` and broadcast the action."""
    try:
        ledger.log_change(agent_id, -abs(ip), -abs(du), "moderation_penalty")
    except Exception:  # pragma: no cover - defensive
        logger.error("Failed to apply penalty to %s", agent_id, exc_info=True)
        return
    event = {
        "type": "moderation",
        "action": "penalty",
        "agent_id": agent_id,
        "ip": ip,
        "du": du,
        "step": sim.current_step,
    }
    await sim.event_kernel.emit_environment_event(event)


__all__ = ["apply_penalty", "mute_agent", "reset_memory"]
