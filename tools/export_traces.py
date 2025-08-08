#!/usr/bin/env python3
"""Generate plots from evaluation events in trace datasets."""

from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export evaluation plots from traces")
    parser.add_argument("traces", help="Path to JSONL trace file")
    parser.add_argument("--outdir", default="plots", help="Directory to store plots")
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

    return 0


if __name__ == "__main__":  # pragma: no cover - manual tool
    raise SystemExit(main())

