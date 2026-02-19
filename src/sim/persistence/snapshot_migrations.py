from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

CURRENT_SNAPSHOT_SCHEMA_VERSION = 3


def _migrate_v1_to_v2(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Normalize world timing fields into ``environment_state``."""

    migrated = deepcopy(snapshot)
    env = migrated.get("environment_state")
    if not isinstance(env, dict):
        env = {}
    env["world_tick"] = int(env.get("world_tick", migrated.get("world_tick", -1)))
    env["world_hour"] = int(env.get("world_hour", migrated.get("world_hour", 0)))
    env["world_day"] = int(env.get("world_day", migrated.get("world_day", 0)))
    world_season = env.get("world_season", migrated.get("world_season"))
    env["world_season"] = int(world_season) if world_season is not None else None
    migrated["environment_state"] = env
    migrated["snapshot_schema_version"] = 2
    return migrated


def _migrate_v2_to_v3(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Backfill environment/agent defaults required by the current schema."""

    migrated = deepcopy(snapshot)

    env = migrated.get("environment_state")
    if not isinstance(env, dict):
        env = {}
    env.setdefault("weather", "clear")
    env.setdefault("season_effects", {})
    env.setdefault("active_global_modifiers", [])
    env.setdefault("council_window_active", False)
    migrated["environment_state"] = env

    agents = migrated.get("agents")
    if isinstance(agents, list):
        for agent_data in agents:
            if not isinstance(agent_data, dict):
                continue
            agent_data.setdefault("lifecycle_state", "active")
            agent_data.setdefault("lifecycle_history", [])
            agent_data.setdefault("legacy_artifacts", {})
            agent_data.setdefault("memory_archival_policy", {})
            agent_data.setdefault("predecessor_id", None)
            agent_data.setdefault("successor_id", None)

    if not isinstance(migrated.get("knowledge_board"), dict):
        migrated["knowledge_board"] = {"entries": [], "vector": {}}
    if not isinstance(migrated.get("world_map"), dict):
        migrated["world_map"] = {"width": 10, "height": 10, "agents": {}, "vector": {}}

    migrated["snapshot_schema_version"] = CURRENT_SNAPSHOT_SCHEMA_VERSION
    return migrated


SNAPSHOT_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
}


def migrate_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Run ordered N->N+1 migrations to normalize snapshot payloads."""

    version_raw = snapshot.get("snapshot_schema_version", 1)
    if not isinstance(version_raw, int):
        raise ValueError("snapshot_schema_version must be an integer")
    if version_raw < 1:
        raise ValueError("snapshot_schema_version must be >= 1")
    if version_raw > CURRENT_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported snapshot schema version {version_raw}; "
            f"maximum supported is {CURRENT_SNAPSHOT_SCHEMA_VERSION}"
        )

    migrated = deepcopy(snapshot)
    version = version_raw
    while version < CURRENT_SNAPSHOT_SCHEMA_VERSION:
        migration = SNAPSHOT_MIGRATIONS.get(version)
        if migration is None:
            raise ValueError(f"Missing migration for snapshot schema version {version}")
        migrated = migration(migrated)
        next_version = migrated.get("snapshot_schema_version")
        if next_version != version + 1:
            raise ValueError(
                "Snapshot migration contract violated: "
                f"expected version {version + 1}, got {next_version}"
            )
        version = next_version

    return migrated
