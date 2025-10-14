#!/usr/bin/env python3
"""CLI to replay simulation snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.infra import event_log
from src.infra.event_log import (
    resolve_replay_event_log as _resolve_event_log,
)
from src.sim.simulation import Simulation


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
        help=(
            "Path to the event log file. If omitted, a log located next to the "
            "snapshot will be used when available."
        ),
    )
    args = parser.parse_args(argv)

    snapshot = Path(args.snapshot)
    events_path = _resolve_event_log(snapshot, args.events)

    seed = args.seed
    if seed is None:
        seed = event_log.get_seed(events_path)

    Simulation.replay_from_snapshot(
        snapshot,
        start_step=args.start,
        end_step=args.end,
        seed=seed,
        events_path=events_path,

    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
