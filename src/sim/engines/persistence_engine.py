from __future__ import annotations

from pathlib import Path
from typing import Any


class PersistenceEngine:
    """Encapsulates snapshot/replay/checkpointing entrypoints."""

    def from_snapshot(
        self, simulation_cls: type[Any], snapshot: dict[str, Any], seed: int | None = None
    ) -> Any:
        return simulation_cls._from_snapshot_impl(snapshot, seed=seed)

    def replay_from_snapshot(
        self,
        simulation_cls: type[Any],
        snapshot_path: str | Path,
        *,
        seed: int | None = None,
        stop_step: int | None = None,
    ) -> Any:
        return simulation_cls._replay_from_snapshot_impl(
            snapshot_path,
            seed=seed,
            stop_step=stop_step,
        )

    def capture_tick(self, simulation: Any, tick: Any) -> dict[str, Any]:
        return {
            "step": tick.step,
            "trace_hash": getattr(simulation, "_last_trace_hash", ""),
            "replay_metadata": dict(tick.replay_metadata),
        }
