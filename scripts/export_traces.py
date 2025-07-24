#!/usr/bin/env python3
"""Export snapshots or Redpanda event logs to a JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.infra import event_log
from src.infra.snapshot import load_snapshot


def export_latest(directory: str | Path = "snapshots", output: str | Path = "data/traces.jsonl") -> Path:
    """Write a dataset from the newest snapshots in ``directory``.

    The snapshots are sorted by step number and all available files are exported
    to ``output`` in JSON Lines format.
    """

    path = Path(directory)
    files = sorted(path.glob("snapshot_*.json*"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"No snapshots found in {directory}")

    out_path = Path(output)
    with out_path.open("w", encoding="utf-8") as out:
        for file in files:
            step = int(file.stem.split("_")[1])
            compress = file.suffix == ".zst"
            snap = load_snapshot(step, directory=directory, compress=compress)
            out.write(json.dumps(snap))
            out.write("\n")

    return out_path


def iter_snapshots(directory: str | Path) -> Iterable[dict[str, Any]]:
    """Yield snapshots from ``directory`` in step order."""
    path = Path(directory)
    files = sorted(path.glob("snapshot_*.json*"), key=lambda p: int(p.stem.split("_")[1]))
    for file in files:
        step = int(file.stem.split("_")[1])
        compress = file.suffix == ".zst"
        yield load_snapshot(step, directory=directory, compress=compress)


def iter_events(
    from_redpanda: bool = False, file: str | Path | None = None
) -> Iterable[dict[str, Any]]:
    """Yield events from Redpanda or a JSON file."""
    if from_redpanda:
        yield from event_log.stream_events(after_step=0, timeout=1.0)
        return

    if file is None:
        raise ValueError("Event file path required when not using --redpanda")

    with Path(file).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    events = data if isinstance(data, list) else data.get("events", [])
    yield from events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export traces to JSONL dataset")
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--snapshots", help="Snapshot directory to read")
    src_group.add_argument("--events", help="JSON file containing event log")
    src_group.add_argument("--redpanda", action="store_true", help="Fetch events from Redpanda")
    parser.add_argument("-o", "--output", help="Output JSONL file (default: stdout)")
    parser.add_argument("--agent", help="Only include events for this agent_id")
    parser.add_argument("--start-step", type=int, help="First step to include (inclusive)")
    parser.add_argument("--end-step", type=int, help="Last step to include (inclusive)")
    args = parser.parse_args(argv)

    if args.snapshots:
        iterator = iter_snapshots(args.snapshots)
    elif args.redpanda:
        iterator = iter_events(from_redpanda=True)
    else:
        iterator = iter_events(file=args.events)

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    for item in iterator:
        step = item.get("step")
        if step is not None:
            if args.start_step is not None and step < args.start_step:
                continue
            if args.end_step is not None and step > args.end_step:
                continue
        if args.agent and item.get("agent_id") != args.agent:
            continue
        out.write(json.dumps(item))
        out.write("\n")
    if args.output:
        out.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
