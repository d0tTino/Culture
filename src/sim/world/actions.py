from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.sim.world.state import WorldState


@dataclass(frozen=True)
class ActionRuleResult:
    allowed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class ActionRulesEngine:
    _WEATHER_ACTION_ALLOWLIST: ClassVar[dict[str, set[str]]] = {
        "storm": {"idle", "propose_idea", "request_role_change"},
    }

    def check(self, *, world_state: WorldState, action: str, actor_id: str) -> ActionRuleResult:
        weather = world_state.environment.weather
        blocked = self._WEATHER_ACTION_ALLOWLIST.get(weather)
        if blocked is not None and action not in blocked:
            return ActionRuleResult(
                allowed=False,
                reason=f"Action '{action}' blocked by weather '{weather}'",
            )

        if action == "build":
            bag = world_state.resources.agent_inventories.get(actor_id, {})
            if bag.get("wood", 0) < 1:
                return ActionRuleResult(False, "Need at least 1 wood to build")
        return ActionRuleResult(True)

    def resource_multiplier(self, *, world_state: WorldState, action: str) -> float:
        weather = world_state.environment.weather
        season = world_state.temporal.world_season
        weather_mult = {
            "clear": {"gather": 1.0, "build": 1.0},
            "rain": {"gather": 0.9, "build": 0.95},
            "windy": {"gather": 0.95, "build": 0.9},
            "storm": {"gather": 0.75, "build": 0.8},
        }
        season_names = ("spring", "summer", "autumn", "winter")
        season_name = season_names[season % len(season_names)] if season is not None else "spring"
        season_mult = {
            "spring": {"gather": 1.1},
            "summer": {"build": 1.05},
            "autumn": {"gather": 1.0},
            "winter": {"gather": 0.8, "build": 0.9},
        }
        return float(weather_mult.get(weather, {}).get(action, 1.0)) * float(
            season_mult.get(season_name, {}).get(action, 1.0)
        )
