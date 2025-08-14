#!/usr/bin/env python3
"""Generate plots from evaluation events and bundle runs for replay."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_metrics(file: str | Path) -> dict[str, list[tuple[int, float]]]:
    data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with Path(file).open("r", encoding="utf-8") as fh:
        for line in fh:
            obj: dict[str, Any] = json.loads(line)
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


def bundle_replay(trace_file: str | Path, bundle: str | Path) -> Path:
    """Package logs, metrics, scenario metrics, and RNG seed into a single archive.

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
    metrics = load_metrics(trace_path)
    scenario_metrics = {
        key: metrics.get(key, [])
        for key in ("coalitions", "sentiment", "collective_du", "collective_ip")
    }
    seed: int | None = None
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if "seed" in obj:
                seed = obj["seed"]
                break

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        shutil.copy(trace_path, tmp / "logs.jsonl")
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
    args = parser.parse_args(argv)

    metrics = load_metrics(args.traces)
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
        bundle_replay(args.traces, args.bundle)

    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())
