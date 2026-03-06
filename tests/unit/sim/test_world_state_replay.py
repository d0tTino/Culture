import asyncio
import json

from src.sim.environment import EnvironmentReducerEvent, EnvironmentSystem
from src.sim.persistence.trace_hash_service import TraceHashService
from src.sim.world import WorldState
from src.sim.world_map import ResourceToken, WorldMap


def _build_environment(world_state: WorldState) -> EnvironmentSystem:
    return EnvironmentSystem(
        world_ticks_per_day=24,
        turns_per_world_tick=1,
        world_season_length_days=3,
        weather_shift_interval_ticks=2,
    )


def test_world_state_replay_is_deterministic_and_serializable() -> None:
    initial = WorldState()
    world_map = WorldMap(width=8, height=8, world_state=initial)
    initial.temporal.world_season = 0
    env = _build_environment(initial)

    asyncio.run(world_map.add_agent("a", x=0, y=0))
    asyncio.run(world_map.add_resource(1, 0, ResourceToken.WOOD, 3))
    events_run_1: list[list[dict[str, object]]] = []
    for turn in range(1, 7):
        initial, events = env.reduce(initial, EnvironmentReducerEvent(turn_index=turn))
        events_run_1.append([event.as_event() for event in events])
        asyncio.run(world_map.move_to("a", 1, 0))
        asyncio.run(world_map.gather("a", ResourceToken.WOOD))

    snapshot = initial.snapshot()
    blob = json.dumps(snapshot, sort_keys=True)
    restored = WorldState.from_snapshot(json.loads(blob))

    replay_map = WorldMap(width=8, height=8, world_state=restored)
    replay_env = _build_environment(restored)
    events_run_2: list[list[dict[str, object]]] = []
    for turn in range(1, 7):
        restored, events = replay_env.reduce(restored, EnvironmentReducerEvent(turn_index=turn))
        events_run_2.append([event.as_event() for event in events])

    assert snapshot == restored.snapshot()
    assert events_run_1 == events_run_2
    assert replay_map.agent_positions["a"] == (1, 0)
    assert TraceHashService.compute(snapshot) == TraceHashService.compute(restored.snapshot())
