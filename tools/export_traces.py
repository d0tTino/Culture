#!/usr/bin/env python3
"""Generate plots from evaluation events and bundle runs for replay."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Optional scenario-level metrics to include when bundling replay artifacts.
scenario_metrics: dict[str, Any] = {}


def iter_events(
    file: str | Path, start: int = 0, end: int | None = None
) -> Iterator[dict[str, Any]]:
    """Yield events from ``file`` within the given tick range."""
    with Path(file).open("r", encoding="utf-8") as fh:
        for line in fh:
            obj: dict[str, Any] = json.loads(line)
            tick = int(obj.get("tick", obj.get("step", 0)))
            if tick < start:
                continue
            if end is not None and tick > end:
                break
            yield obj


def load_metrics(
    file: str | Path, start: int = 0, end: int | None = None
) -> dict[str, list[tuple[int, float]]]:
    data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for obj in iter_events(file, start, end):
        if obj.get("type") != "evaluation":
            continue
        step = obj.get("step")
        if not isinstance(step, int):
            continue
        for key, value in obj.items():
            if key in {"type", "step", "trace_hash"}:
                continue
            if isinstance(value, (int, float)):
                data[key].append((step, float(value)))
    return data


def bundle_replay(
    trace_file: str | Path,
    bundle: str | Path,
    start: int = 0,
    end: int | None = None,
) -> Path:
    """Package sliced logs, metrics and RNG seed into a single archive.


    Parameters
    ----------
    trace_file:
        Path to the JSONL trace log.
    bundle:
        Output archive path without extension.

    Returns
    -------
    Path
        Path to the created ``.tar.gz`` archive.
    """
    trace_path = Path(trace_file)
    metrics = load_metrics(trace_path, start, end)

    seed: int | None = None
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        logs_path = tmp / "logs.jsonl"
        with logs_path.open("w", encoding="utf-8") as out:
            for obj in iter_events(trace_path, start, end):
                if seed is None and "seed" in obj:
                    seed = obj["seed"]
                out.write(json.dumps(obj))
                out.write("\n")
        with (tmp / "metrics.json").open("w", encoding="utf-8") as mfh:
            json.dump(metrics, mfh)
        with (tmp / "scenario_metrics.json").open("w", encoding="utf-8") as smfh:
            json.dump(scenario_metrics, smfh)
        if seed is not None:
            with (tmp / "seed.txt").open("w", encoding="utf-8") as sfh:
                sfh.write(str(seed))
        archive = shutil.make_archive(str(Path(bundle)), "gztar", tmp)
    return Path(archive)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export evaluation plots and bundle artifacts for replay"
    )
    parser.add_argument("traces", help="Path to JSONL trace file")
    parser.add_argument("--outdir", default="plots", help="Directory to store plots")
    parser.add_argument("--bundle", help="Output path for replay bundle (without extension)")
    parser.add_argument("--start", type=int, default=0, help="First tick to include")
    parser.add_argument("--end", type=int, help="Last tick to include")
    args = parser.parse_args(argv)

    metrics = load_metrics(args.traces, args.start, args.end)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for metric, pairs in metrics.items():
        if not pairs:
            continue
        pairs.sort()
        steps, values = zip(*pairs)
        plt.figure()
        plt.plot(steps, values)
        plt.xlabel("Step")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(metric.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(outdir / f"{metric}.png")
        plt.close()

    if args.bundle:
        bundle_replay(args.traces, args.bundle, args.start, args.end)

    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
