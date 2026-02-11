# Replay and Snapshot Migration Notes: Tick-Indexed World Time

## What changed

Simulation world time is now advanced on a **world tick boundary** (`world_tick`) instead of advancing once per agent turn. A configurable turn quantum (`WORLD_TICK_TURN_QUANTUM`) determines how many agent turns map to one world tick.

Agent action events now include both:

- `turn_index`: the per-agent-turn progression index (legacy `step` semantics), and
- `world_time`: a snapshot object (`world_tick`, `world_hour`, `world_day`, `world_season`, `formatted`).

Snapshots now persist:

- `world_tick`, and
- `turns_per_world_tick`.

## Backward compatibility behavior

Historical traces and snapshots remain readable:

1. **Legacy snapshots without `world_tick`**
   - Replay derives `world_tick` from `step` using configured or snapshot-derived `turns_per_world_tick`.
2. **Legacy events without `world_time` object**
   - Replay falls back to legacy top-level `world_hour/world_day/world_season` fields when present.
3. **Legacy action payloads without `turn_index`**
   - `step` continues to be accepted as the turn index.

## Interpreting mixed-era traces

When inspecting old and new events together:

- Use `turn_index` if present, otherwise fall back to `step`.
- Use `world_time.world_tick` if present for cadence and day-boundary logic.
- If `world_time` is absent, interpret time using the old top-level world fields and treat cadence assumptions as turn-based for those events.
