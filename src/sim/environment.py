from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvironmentState:
    world_tick: int = 0
    world_hour: int = 0
    world_day: int = 0
    world_season: int | None = None
    weather: str = "clear"
    season_effects: dict[str, Any] = field(default_factory=dict)
    active_global_modifiers: list[str] = field(default_factory=list)
    council_window_active: bool = False


class EnvironmentSystem:
    """Owns temporal progression and environment context generation."""

    _WEATHER_CYCLE = ("clear", "rain", "windy", "storm")
    _SEASON_NAMES = ("spring", "summer", "autumn", "winter")

    def __init__(
        self,
        *,
        state: EnvironmentState,
        world_ticks_per_day: int,
        turns_per_world_tick: int,
        world_season_length_days: int | None,
        weather_shift_interval_ticks: int = 6,
        council_window_days: int = 7,
        council_window_start_hour: int = 9,
        council_window_duration_hours: int = 3,
        world_time_broadcast_cadence_ticks: int = 24,
    ) -> None:
        self.state = state
        self.world_ticks_per_day = max(1, int(world_ticks_per_day))
        self.turns_per_world_tick = max(1, int(turns_per_world_tick))
        self.world_season_length_days = (
            int(world_season_length_days) if world_season_length_days is not None else None
        )
        self.weather_shift_interval_ticks = max(1, int(weather_shift_interval_ticks))
        self.council_window_days = max(1, int(council_window_days))
        self.council_window_start_hour = max(0, int(council_window_start_hour))
        self.council_window_duration_hours = max(1, int(council_window_duration_hours))
        self.world_time_broadcast_cadence_ticks = max(1, int(world_time_broadcast_cadence_ticks))

    def _format_world_time(self) -> str:
        time_str = f"Day {self.state.world_day}, {self.state.world_hour:02d}:00"
        if self.state.world_season is not None:
            time_str += f" (Season {self.state.world_season})"
        return time_str

    def _season_name(self) -> str | None:
        if self.state.world_season is None:
            return None
        return self._SEASON_NAMES[self.state.world_season % len(self._SEASON_NAMES)]

    def _condition_hooks(self) -> dict[str, Any]:
        weather_hooks: dict[str, dict[str, Any]] = {
            "clear": {
                "resource_multipliers": {"gather": 1.0, "build": 1.0},
                "allowed_actions": ["*"],
                "mood_modifiers": {"baseline": 0.05},
            },
            "rain": {
                "resource_multipliers": {"gather": 0.9, "build": 0.95},
                "allowed_actions": ["*"],
                "mood_modifiers": {"baseline": -0.02},
            },
            "windy": {
                "resource_multipliers": {"gather": 0.95, "build": 0.9},
                "allowed_actions": ["*"],
                "mood_modifiers": {"baseline": 0.0},
            },
            "storm": {
                "resource_multipliers": {"gather": 0.75, "build": 0.8},
                "allowed_actions": ["idle", "propose_idea", "request_role_change"],
                "mood_modifiers": {"baseline": -0.1},
            },
        }
        season_hooks: dict[str, dict[str, Any]] = {
            "spring": {
                "resource_multipliers": {"gather": 1.1},
                "mood_modifiers": {"baseline": 0.05},
            },
            "summer": {
                "resource_multipliers": {"build": 1.05},
                "mood_modifiers": {"baseline": 0.02},
            },
            "autumn": {
                "resource_multipliers": {"gather": 1.0},
                "mood_modifiers": {"baseline": 0.0},
            },
            "winter": {
                "resource_multipliers": {"gather": 0.8, "build": 0.9},
                "mood_modifiers": {"baseline": -0.06},
            },
        }
        weather = weather_hooks.get(self.state.weather, weather_hooks["clear"])
        season = season_hooks.get(self._season_name() or "spring", {})
        resource = {
            **weather.get("resource_multipliers", {}),
            **season.get("resource_multipliers", {}),
        }
        mood = {**weather.get("mood_modifiers", {}), **season.get("mood_modifiers", {})}
        return {
            "resource_multipliers": resource,
            "allowed_actions": list(weather.get("allowed_actions", ["*"])),
            "mood_modifiers": mood,
        }

    def world_time_snapshot(self) -> dict[str, Any]:
        return {
            "world_tick": self.state.world_tick,
            "world_hour": self.state.world_hour,
            "world_day": self.state.world_day,
            "world_season": self.state.world_season,
            "formatted": self._format_world_time(),
        }

    def build_perception_context(self, *, turn_index: int) -> dict[str, Any]:
        return {
            "turn_index": turn_index,
            "time": self.world_time_snapshot(),
            "weather": self.state.weather,
            "season": self._season_name(),
            "season_effects": dict(self.state.season_effects),
            "active_global_modifiers": list(self.state.active_global_modifiers),
            "council_window_active": self.state.council_window_active,
            "effect_hooks": self._condition_hooks(),
        }

    def tick(self, turn_index: int) -> list[dict[str, Any]]:
        if turn_index <= 0:
            return []

        target_tick = (turn_index - 1) // self.turns_per_world_tick
        events: list[dict[str, Any]] = []
        while self.state.world_tick < target_tick:
            self.state.world_tick += 1
            self.state.world_hour += 1
            if self.state.world_hour >= self.world_ticks_per_day:
                self.state.world_hour = 0
                self.state.world_day += 1
                events.append(self._event("daily_reset", turn_index))

            if self.state.world_tick % self.weather_shift_interval_ticks == 0:
                idx = (self.state.world_tick // self.weather_shift_interval_ticks) % len(
                    self._WEATHER_CYCLE
                )
                self.state.weather = self._WEATHER_CYCLE[idx]
                events.append(self._event("weather_shift", turn_index))

            if self.world_season_length_days and self.state.world_day > 0:
                new_season = self.state.world_day // self.world_season_length_days
                if new_season != self.state.world_season:
                    self.state.world_season = new_season
                    self.state.season_effects = {
                        "season": self._season_name(),
                        "hooks": self._condition_hooks(),
                    }
                    events.append(self._event("season_transition", turn_index))

            is_council_day = (self.state.world_day % self.council_window_days) == 0
            in_hour_window = (
                self.council_window_start_hour
                <= self.state.world_hour
                < self.council_window_start_hour + self.council_window_duration_hours
            )
            council_active = is_council_day and in_hour_window
            if council_active != self.state.council_window_active:
                self.state.council_window_active = council_active
                events.append(self._event("council_meeting_window", turn_index))

            if self.state.world_tick % self.world_time_broadcast_cadence_ticks == 0:
                events.append(self._event("world_time", turn_index))

        return events

    def _event(self, event_name: str, turn_index: int) -> dict[str, Any]:
        return {
            "type": "environment",
            "event_name": event_name,
            "turn_index": turn_index,
            "world_time": self.world_time_snapshot(),
            "weather": self.state.weather,
            "season": self._season_name(),
            "council_window_active": self.state.council_window_active,
            "active_global_modifiers": list(self.state.active_global_modifiers),
            "effect_hooks": self._condition_hooks(),
        }
