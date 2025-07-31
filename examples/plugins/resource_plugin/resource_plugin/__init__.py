"""Plug-in adding a custom crystal gathering action."""

from __future__ import annotations

from typing import Any

from src.extensions import PluginResult, register_map_action
from src.infra import config
from src.infra.ledger import log_reward, run_auction


async def gather_crystal(
    sim: Any,
    agent_index: int,
    agent_id: str,
    state: Any,
    action: dict[str, Any],
) -> dict[str, Any]:
    """Gather a crystal from the agent's current position."""
    pos = sim.world_map.agent_positions.get(agent_id)
    success = False
    if pos is not None:
        cell = sim.world_map.resources.get(pos)
        if cell and cell.get("crystal", 0) > 0:
            cell["crystal"] -= 1
            if cell["crystal"] == 0:
                del cell["crystal"]
            bag = sim.world_map.agent_resources.setdefault(agent_id, {})
            bag["crystal"] = bag.get("crystal", 0) + 1
            try:
                from src.infra.ledger import ledger

                ledger.add_tokens(agent_id, "crystal", 1)
            except Exception:  # pragma: no cover - optional
                pass
            sim.world_map.vector.increment(agent_id)
            success = True
            start_ip = state.ip
            if config.MAP_GATHER_DU_COST > 0:
                run_auction("gather_crystal", agent_id, config.MAP_GATHER_DU_COST)
                state.du -= config.MAP_GATHER_DU_COST
            start_du = state.du
            state.ip -= config.MAP_GATHER_IP_COST
            state.ip += config.MAP_GATHER_IP_REWARD
            state.du += config.MAP_GATHER_DU_REWARD
            log_reward(
                agent_id,
                state.ip - start_ip,
                state.du - start_du,
                "gather_crystal",
            )
    return {"resource": "crystal", "success": success}


def setup() -> PluginResult:
    """Entry point for :func:`load_plugins`."""
    register_map_action("gather_crystal", gather_crystal)
    return None
