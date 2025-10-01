#!/usr/bin/env python3
"""CLI to replay simulation snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from src.infra import event_log
from src.sim.simulation import Simulation


def _candidate_event_logs(snapshot: Path) -> Iterable[Path]:
    """Yield plausible event log paths relative to ``snapshot``."""

    directory = snapshot.parent
    stem = snapshot.stem
    if stem.endswith(".json"):
        stem = stem[:-5]
    candidates = [
        directory / "events.jsonl",
        directory / "event_log.jsonl",
        directory / f"{stem}.events.jsonl",
        directory / f"{stem}.event_log.jsonl",
        snapshot.with_suffix(".jsonl"),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _resolve_event_log(snapshot: Path, explicit: str | None) -> Path | None:
    """Determine which event log file should be used for replay."""

    if explicit:
        return Path(explicit)
    for candidate in _candidate_event_logs(snapshot):
        if candidate.exists():
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
