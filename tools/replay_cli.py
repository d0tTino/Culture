#!/usr/bin/env python3
"""CLI to replay simulation snapshots."""

from __future__ import annotations

import argparse

from src.infra import event_log
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
    args = parser.parse_args(argv)

    seed = args.seed
    if seed is None:
        seed = event_log.get_seed()

    Simulation.replay_from_snapshot(
        args.snapshot, start_step=args.start, end_step=args.end, seed=seed
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
