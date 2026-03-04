from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from src.extensions import MAP_ACTION_REGISTRY
from src.infra import config
from src.infra.ledger import log_reward, run_auction
from src.interfaces.dashboard_backend import emit_map_change_event
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
    """Handle a map action for an agent and update the simulation state.

    Supported actions include ``move``, ``gather`` and ``build``. The
    function applies the requested action to the ``world_map`` and updates
    the agent's ``current_state`` accordingly. A dashboard event is emitted
    and the agent's updated state is persisted.

    Parameters
    ----------
    sim:
        The active :class:`~src.sim.simulation.Simulation` instance.
    agent_index:
        Index of the agent within ``sim.agents`` whose state should be
        updated.
    agent_id:
        Unique identifier for the agent performing the action.
    current_state:
        The mutable state of the agent prior to applying the action.
    map_action:
        Dictionary describing the requested action and its parameters.
    """
    action_type = map_action.get("action")
    details: dict[str, Any] = {}
    if isinstance(action_type, str) and hasattr(sim, "action_rules_engine"):
        decision = sim.action_rules_engine.check(
            world_state=sim.world_state,
            action=action_type,
            actor_id=agent_id,
        )
        if not decision.allowed:
            details = {"blocked": True, "reason": decision.reason}
            action_type = "idle"
    if action_type == "move":
        if "x" in map_action and "y" in map_action:
            tx = int(map_action.get("x", 0))
            ty = int(map_action.get("y", 0))
            pos = await sim.world_map.move_to(agent_id, tx, ty, vector=sim.vector.to_dict())
        else:
            dx = int(map_action.get("dx", 0))
            dy = int(map_action.get("dy", 0))
            pos = await sim.world_map.move(agent_id, dx, dy, vector=sim.vector.to_dict())
            details = {"position": pos}
            start_ip = current_state.ip
            if config.MAP_MOVE_DU_COST > 0:
                run_auction("move", agent_id, config.MAP_MOVE_DU_COST)
                current_state.du -= config.MAP_MOVE_DU_COST
            start_du = current_state.du
            current_state.ip -= config.MAP_MOVE_IP_COST
            current_state.ip += config.MAP_MOVE_IP_REWARD
            current_state.du += config.MAP_MOVE_DU_REWARD
            log_reward(
                agent_id,
                current_state.ip - start_ip,
                current_state.du - start_du,
                "move",
            )
    elif action_type == "gather":
        res = map_action.get("resource")
        success = False
        if isinstance(res, str):
            success = await sim.world_map.gather(
                agent_id,
                ResourceToken(res),
                vector=sim.vector.to_dict(),
            )
            details = {"resource": res, "success": success}
            if success:
                start_ip = current_state.ip
                if config.MAP_GATHER_DU_COST > 0:
                    run_auction("gather", agent_id, config.MAP_GATHER_DU_COST)
                    current_state.du -= config.MAP_GATHER_DU_COST
                start_du = current_state.du
                mult = (
                    sim.action_rules_engine.resource_multiplier(
                        world_state=sim.world_state, action="gather"
                    )
                    if hasattr(sim, "action_rules_engine")
                    else 1.0
                )
                current_state.ip -= config.MAP_GATHER_IP_COST
                current_state.ip += config.MAP_GATHER_IP_REWARD * mult
                current_state.du += config.MAP_GATHER_DU_REWARD * mult
                log_reward(
                    agent_id,
                    current_state.ip - start_ip,
                    current_state.du - start_du,
                    "gather",
                )
    elif action_type == "build":
        struct = map_action.get("structure")
        success = False
        if isinstance(struct, str):
            success = await sim.world_map.build(
                agent_id,
                StructureType(struct),
                vector=sim.vector.to_dict(),
            )
            details = {"structure": struct, "success": success}
            if success:
                start_ip = current_state.ip
                if config.MAP_BUILD_DU_COST > 0:
                    run_auction("build", agent_id, config.MAP_BUILD_DU_COST)
                    current_state.du -= config.MAP_BUILD_DU_COST
                start_du = current_state.du
                mult = (
                    sim.action_rules_engine.resource_multiplier(
                        world_state=sim.world_state, action="build"
                    )
                    if hasattr(sim, "action_rules_engine")
                    else 1.0
                )
                current_state.ip -= config.MAP_BUILD_IP_COST
                current_state.ip += config.MAP_BUILD_IP_REWARD * mult
                current_state.du += config.MAP_BUILD_DU_REWARD * mult
                log_reward(
                    agent_id,
                    current_state.ip - start_ip,
                    current_state.du - start_du,
                    "build",
                )
        else:
            details = {}
    else:
        custom = await MAP_ACTION_REGISTRY.run(
            action_type,
            sim,
            agent_index,
            agent_id,
            current_state,
            map_action,
        )
        if custom is None:
            return
        details = custom or {}

    map_event_data = {
        "type": "map_action",
        "agent_id": agent_id,
        "step": sim.current_step,
        "action": action_type,
        **details,
    }
    await sim.event_kernel.schedule_immediate(
        lambda data=map_event_data: sim.event_kernel.emit_environment_event(data),
        vector=sim.vector,
    )
    await emit_map_change_event(sim.world_map.to_dict())
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
