#!/usr/bin/env python3
"""CLI to replay simulation snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.infra import event_log
from src.sim.simulation import Simulation


def _discover_event_log(snapshot_path: str | Path) -> Path | None:
    """Infer the event log location from the snapshot hierarchy."""

    path = Path(snapshot_path)
    for directory in path.parents:
        candidate = directory / "event_log.jsonl"
        if candidate.is_file():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a simulation snapshot")
    parser.add_argument("snapshot", help="Path to the snapshot file")
    parser.add_argument(
        "--from-tick",
        "--from",
        dest="start",
        type=int,
        required=True,
        help="First tick to replay",
    )
    parser.add_argument(
        "--to-tick",
        "--to",
        dest="end",
        type=int,
        required=True,
        help="Last tick to replay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override RNG seed (defaults to seed from event log)",
    )
    parser.add_argument(
        "--events",
        type=Path,
        help="Path to the event log (auto-detected relative to the snapshot if omitted)",
    )
    args = parser.parse_args(argv)

    event_log_path: Path | None
    if args.events is not None:
        if not args.events.is_file():
            parser.error(f"Event log not found: {args.events}")
        event_log_path = args.events
    else:
        event_log_path = _discover_event_log(args.snapshot)

    seed = args.seed
    if seed is None:
        seed = event_log.get_seed(path=event_log_path)

    Simulation.replay_from_snapshot(
        args.snapshot,
        start_step=args.start,
        end_step=args.end,
        seed=seed,
        event_log_path=event_log_path,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
