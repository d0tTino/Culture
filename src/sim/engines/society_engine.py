from __future__ import annotations

from typing import Any

from src.sim.contracts.tick_context import TickContext


class SocietyEngine:
    """Keeps society/governance/project lifecycle coordination isolated."""

    def snapshot(self, simulation: Any, tick: TickContext) -> dict[str, Any]:
        return {
            "project_count": len(getattr(simulation, "projects", {})),
            "council_window_active": tick.governance_state.get("council_window_active", False),
        }
