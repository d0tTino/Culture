from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.sim.world.state import EnvironmentTickDelta, WorldState


@dataclass(frozen=True)
class EnvironmentReducerEvent:
    turn_index: int


@dataclass(frozen=True)
class EnvironmentConfig:
    world_ticks_per_day: int
    turns_per_world_tick: int
    world_season_length_days: int | None
    weather_shift_interval_ticks: int
    council_window_days: int
    council_window_start_hour: int
    council_window_duration_hours: int
    world_time_broadcast_cadence_ticks: int


class EnvironmentSystem:
    """Pure world-environment reducers and read projections."""

    _WEATHER_CYCLE = ("clear", "rain", "windy", "storm")
    _SEASON_NAMES = ("spring", "summer", "autumn", "winter")

    def __init__(
        self,
        *,
        world_ticks_per_day: int,
        turns_per_world_tick: int,
        world_season_length_days: int | None,
        weather_shift_interval_ticks: int = 6,
        council_window_days: int = 7,
        council_window_start_hour: int = 9,
        council_window_duration_hours: int = 3,
        world_time_broadcast_cadence_ticks: int = 24,
    ) -> None:
        self.config = EnvironmentConfig(
            world_ticks_per_day=max(1, int(world_ticks_per_day)),
            turns_per_world_tick=max(1, int(turns_per_world_tick)),
            world_season_length_days=(
                int(world_season_length_days) if world_season_length_days is not None else None
            ),
            weather_shift_interval_ticks=max(1, int(weather_shift_interval_ticks)),
            council_window_days=max(1, int(council_window_days)),
            council_window_start_hour=max(0, int(council_window_start_hour)),
            council_window_duration_hours=max(1, int(council_window_duration_hours)),
            world_time_broadcast_cadence_ticks=max(1, int(world_time_broadcast_cadence_ticks)),
        )

    def set_turns_per_world_tick(self, value: int) -> None:
        self.config = EnvironmentConfig(
            **{**self.config.__dict__, "turns_per_world_tick": max(1, int(value))}
        )

    def _format_world_time(self, world_state: WorldState) -> str:
        time_str = (
            f"Day {world_state.temporal.world_day}, "
            f"{world_state.temporal.world_hour:02d}:00"
        )
        if world_state.temporal.world_season is not None:
            time_str += f" (Season {world_state.temporal.world_season})"
        return time_str

    def season_name(self, world_state: WorldState) -> str | None:
        if world_state.temporal.world_season is None:
            return None
        return self._SEASON_NAMES[world_state.temporal.world_season % len(self._SEASON_NAMES)]

    def condition_hooks(self, world_state: WorldState) -> dict[str, Any]:
        # unchanged semantics
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
        weather = weather_hooks.get(world_state.environment.weather, weather_hooks["clear"])
        season = season_hooks.get(self.season_name(world_state) or "spring", {})
        return {
            "resource_multipliers": {
                **weather.get("resource_multipliers", {}),
                **season.get("resource_multipliers", {}),
            },
            "allowed_actions": list(weather.get("allowed_actions", ["*"])),
            "mood_modifiers": {
                **weather.get("mood_modifiers", {}),
                **season.get("mood_modifiers", {}),
            },
        }

    def world_time_snapshot(self, world_state: WorldState) -> dict[str, Any]:
        return {
            "world_tick": world_state.temporal.world_tick,
            "world_hour": world_state.temporal.world_hour,
            "world_day": world_state.temporal.world_day,
            "world_season": world_state.temporal.world_season,
            "formatted": self._format_world_time(world_state),
        }

    def tick(
        self, world_state: WorldState, turn_index: int
    ) -> tuple[WorldState, tuple[EnvironmentTickDelta, ...]]:
        return self.reduce(world_state, EnvironmentReducerEvent(turn_index=turn_index))

    def reduce(
        self, world_state: WorldState, event: EnvironmentReducerEvent
    ) -> tuple[WorldState, tuple[EnvironmentTickDelta, ...]]:
        if event.turn_index <= 0:
            return world_state, ()
        state = WorldState.from_snapshot(world_state.snapshot())
        target_tick = (event.turn_index - 1) // self.config.turns_per_world_tick
        events: list[EnvironmentTickDelta] = []
        while state.temporal.world_tick < target_tick:
            state.temporal.world_tick += 1
            state.temporal.world_hour += 1
            if state.temporal.world_hour >= self.config.world_ticks_per_day:
                state.temporal.world_hour = 0
                state.temporal.world_day += 1
                events.append(self._event(state, "daily_reset", event.turn_index))
            if state.temporal.world_tick % self.config.weather_shift_interval_ticks == 0:
                idx = (state.temporal.world_tick // self.config.weather_shift_interval_ticks) % len(
                    self._WEATHER_CYCLE
                )
                state.environment.weather = self._WEATHER_CYCLE[idx]
                events.append(self._event(state, "weather_shift", event.turn_index))
            if self.config.world_season_length_days and state.temporal.world_day > 0:
                new_season = state.temporal.world_day // self.config.world_season_length_days
                if new_season != state.temporal.world_season:
                    state.temporal.world_season = new_season
                    state.environment.season_effects = {
                        "season": self.season_name(state),
                        "hooks": self.condition_hooks(state),
                    }
                    events.append(self._event(state, "season_transition", event.turn_index))
            is_council_day = (state.temporal.world_day % self.config.council_window_days) == 0
            in_hour_window = (
                self.config.council_window_start_hour
                <= state.temporal.world_hour
                < self.config.council_window_start_hour + self.config.council_window_duration_hours
            )
            council_active = is_council_day and in_hour_window
            if council_active != state.environment.council_window_active:
                state.environment.council_window_active = council_active
                events.append(self._event(state, "council_meeting_window", event.turn_index))
            if state.temporal.world_tick % self.config.world_time_broadcast_cadence_ticks == 0:
                events.append(self._event(state, "world_time", event.turn_index))
        return state, tuple(events)

    def _event(self, world_state: WorldState, event_name: str, turn_index: int) -> EnvironmentTickDelta:
        return EnvironmentTickDelta(
            event_name=event_name,
            turn_index=turn_index,
            world_time=self.world_time_snapshot(world_state),
            weather=world_state.environment.weather,
            season=self.season_name(world_state),
            council_window_active=world_state.environment.council_window_active,
            active_global_modifiers=tuple(world_state.environment.active_global_modifiers),
            effect_hooks=self.condition_hooks(world_state),
        )
