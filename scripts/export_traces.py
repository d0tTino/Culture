#!/usr/bin/env python3
"""Export snapshots or Redpanda event logs to a JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Any

from src.infra.snapshot import load_snapshot
from src.infra import event_log


def iter_snapshots(directory: str | Path) -> Iterable[dict[str, Any]]:
    """Yield snapshots from ``directory`` in step order."""
    path = Path(directory)
    files = sorted(path.glob("snapshot_*.json*"), key=lambda p: int(p.stem.split("_")[1]))
    for file in files:
        step = int(file.stem.split("_")[1])
        compress = file.suffix == ".zst"
        yield load_snapshot(step, directory=directory, compress=compress)


def iter_events(from_redpanda: bool = False, file: str | Path | None = None) -> Iterable[dict[str, Any]]:
    """Yield events from Redpanda or a JSON file."""
    if from_redpanda:
        yield from event_log.stream_events(after_step=0, timeout=1.0)
        return

    if file is None:
        raise ValueError("Event file path required when not using --redpanda")

    with Path(file).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    events = data if isinstance(data, list) else data.get("events", [])
    for event in events:
        yield event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export traces to JSONL dataset")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--snapshots", help="Snapshot directory to read")
    src_group.add_argument("--events", help="JSON file containing event log")
    src_group.add_argument(
        "--redpanda", action="store_true", help="Fetch events from Redpanda"
    )
    parser.add_argument("-o", "--output", help="Output JSONL file (default: stdout)")
    args = parser.parse_args(argv)

    if args.snapshots:
        iterator = iter_snapshots(args.snapshots)
    elif args.redpanda:
        iterator = iter_events(from_redpanda=True)
    else:
        iterator = iter_events(file=args.events)

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    for item in iterator:
        out.write(json.dumps(item))
        out.write("\n")
    if args.output:
        out.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
