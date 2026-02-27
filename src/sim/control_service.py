from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.interfaces.interaction_schema import InteractionContext


class SimulationControlService:
    """Backward-compatible wrapper for control commands."""

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation

    async def handle_control_command(self, cmd: Mapping[str, Any]) -> dict[str, Any] | None:
        result = await self.simulation.command_service.execute_from_payload(
            dict(cmd),
            context=InteractionContext(
                sender_id="simulation",
                source="simulation",
                permissions={"admin", "moderator"},
            ),
        )
        return result.data if isinstance(result.data, dict) else None
