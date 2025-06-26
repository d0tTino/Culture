from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from src.infra import config
from src.sim.world_map import ResourceToken, StructureType

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from src.agents.core.agent_state import AgentState

    from .simulation import Simulation

logger = logging.getLogger(__name__)


async def process_map_action(
    sim: Simulation,
    agent_index: int,
    agent_id: str,
    current_state: AgentState,
    map_action: dict[str, Any],
) -> None:
    """Execute a world map action for the given agent."""
    action_type = map_action.get("action")
    details: dict[str, Any] = {}
    if action_type == "move":
        if "x" in map_action and "y" in map_action:
            tx = int(map_action.get("x", 0))
            ty = int(map_action.get("y", 0))
            pos = sim.world_map.move_to(agent_id, tx, ty, vector=sim.vector.to_dict())
        else:
            dx = int(map_action.get("dx", 0))
            dy = int(map_action.get("dy", 0))
            pos = sim.world_map.move(agent_id, dx, dy, vector=sim.vector.to_dict())
        details = {"position": pos}
        start_ip = current_state.ip
        if config.MAP_MOVE_DU_COST > 0:
            try:
                from src.infra.ledger import ledger

                aid = ledger.open_auction("move")
                ledger.place_bid(aid, agent_id, config.MAP_MOVE_DU_COST)
                ledger.resolve_auction(aid)
            except Exception:  # pragma: no cover - optional
                logger.debug("Ledger auction failed", exc_info=True)
            current_state.du -= config.MAP_MOVE_DU_COST
        start_du = current_state.du
        current_state.ip -= config.MAP_MOVE_IP_COST
        current_state.ip += config.MAP_MOVE_IP_REWARD
        current_state.du += config.MAP_MOVE_DU_REWARD
        try:
            from src.infra.ledger import ledger

            ledger.log_change(
                agent_id,
                current_state.ip - start_ip,
                current_state.du - start_du,
                "move",
            )
        except Exception:  # pragma: no cover - optional
            logger.debug("Ledger logging failed", exc_info=True)
    elif action_type == "gather":
        res = map_action.get("resource")
        success = False
        if isinstance(res, str):
            success = sim.world_map.gather(
                agent_id,
                ResourceToken(res),
                vector=sim.vector.to_dict(),
            )
        details = {"resource": res, "success": success}
        if success:
            start_ip = current_state.ip
            if config.MAP_GATHER_DU_COST > 0:
                try:
                    from src.infra.ledger import ledger

                    aid = ledger.open_auction("gather")
                    ledger.place_bid(aid, agent_id, config.MAP_GATHER_DU_COST)
                    ledger.resolve_auction(aid)
                except Exception:  # pragma: no cover - optional
                    logger.debug("Ledger auction failed", exc_info=True)
                current_state.du -= config.MAP_GATHER_DU_COST
            start_du = current_state.du
            current_state.ip -= config.MAP_GATHER_IP_COST
            current_state.ip += config.MAP_GATHER_IP_REWARD
            current_state.du += config.MAP_GATHER_DU_REWARD
            try:
                from src.infra.ledger import ledger

                ledger.log_change(
                    agent_id,
                    current_state.ip - start_ip,
                    current_state.du - start_du,
                    "gather",
                )
            except Exception:  # pragma: no cover - optional
                logger.debug("Ledger logging failed", exc_info=True)
    elif action_type == "build":
        struct = map_action.get("structure")
        success = False
        if isinstance(struct, str):
            success = sim.world_map.build(
                agent_id,
                StructureType(struct),
                vector=sim.vector.to_dict(),
            )
        details = {"structure": struct, "success": success}
        if success:
            start_ip = current_state.ip
            if config.MAP_BUILD_DU_COST > 0:
                try:
                    from src.infra.ledger import ledger

                    aid = ledger.open_auction("build")
                    ledger.place_bid(aid, agent_id, config.MAP_BUILD_DU_COST)
                    ledger.resolve_auction(aid)
                except Exception:  # pragma: no cover - optional
                    logger.debug("Ledger auction failed", exc_info=True)
                current_state.du -= config.MAP_BUILD_DU_COST
            start_du = current_state.du
            current_state.ip -= config.MAP_BUILD_IP_COST
            current_state.ip += config.MAP_BUILD_IP_REWARD
            current_state.du += config.MAP_BUILD_DU_REWARD
            try:
                from src.infra.ledger import ledger

                ledger.log_change(
                    agent_id,
                    current_state.ip - start_ip,
                    current_state.du - start_du,
                    "build",
                )
            except Exception:  # pragma: no cover - optional
                logger.debug("Ledger logging failed", exc_info=True)
    else:
        details = {}

    map_event_data = {
        "type": "map_action",
        "agent_id": agent_id,
        "step": sim.current_step,
        "action": action_type,
        **details,
    }
    await sim.event_kernel.schedule(
        lambda data=map_event_data: sim._emit_environment_event(data),
        vector=sim.vector,
    )
    if sim.discord_bot:
        embed = sim.discord_bot.create_map_action_embed(
            agent_id=agent_id,
            action=action_type,
            details=details,
            step=sim.current_step,
        )
        task = asyncio.create_task(
            sim.discord_bot.send_simulation_update(embed=embed, agent_id=agent_id)
        )
        _ = task

    sim.agents[agent_index].update_state(current_state)


__all__ = ["process_map_action"]
