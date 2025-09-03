#!/usr/bin/env python3
"""Run the signature_demo scenario and export artifacts."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import yaml

from scripts.export_traces import load_metrics
from src.app import create_simulation

RESULT_DIR = Path("results/signature_demo")
SCENARIO_PATH = Path("scenarios/signature_demo.yaml")


async def main() -> None:
    """Execute the signature_demo scenario and save outputs."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["EVENT_LOG_PATH"] = str(RESULT_DIR / "event_log.jsonl")

    with SCENARIO_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    scenario_desc = data.get("description", "")
    steps = int(data.get("steps", 0) or 0)
    num_agents = int(data.get("agents", 3) or 3)
    beats = data.get("beats") or []
    eval_hooks = data.get("evaluation_hooks") or []

    sim = create_simulation(
        num_agents=num_agents,
        steps=steps,
        scenario=scenario_desc,
        beats=beats,
        seed=42,
    )
    if eval_hooks:
        sim.register_named_evaluation_hooks(list(map(str, eval_hooks)))

    await sim.async_run(steps)

    metrics_path = RESULT_DIR / "metrics.json"
    metrics = load_metrics(RESULT_DIR / "event_log.jsonl")
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Event log: {RESULT_DIR / 'event_log.jsonl'}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())
