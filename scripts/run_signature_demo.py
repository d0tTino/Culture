#!/usr/bin/env python3
"""Run the ``signature_demo`` scenario and collect artifacts."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from scripts import export_traces
from src.app import create_simulation, load_scenario
from src.infra import event_log
from src.infra.checkpoint import capture_rng_state
from src.infra.snapshot import compute_trace_hash, save_snapshot

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from src.sim.simulation import Simulation

RESULT_DIR = Path("results/signature_demo")
SCENARIO_PATH = Path("scenarios/signature_demo.yaml")
README_PATH = RESULT_DIR / "README.md"


def _clean_results_dir() -> None:
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)


def _generate_plots(
    metrics: dict[str, list[tuple[int, float]]], outdir: Path
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return []

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - best effort
        raise RuntimeError("matplotlib is required to generate plots") from exc

    plot_paths: list[Path] = []
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
        path = outdir / f"{metric}.png"
        plt.savefig(path)
        plt.close()
        plot_paths.append(path)
    return plot_paths


def _create_snapshot(sim: Simulation, directory: Path) -> Path:
    from src.sim.simulation import Simulation

    if not isinstance(sim, Simulation):  # pragma: no cover - defensive
        raise TypeError("Expected a Simulation instance")

    directory.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "step": sim.current_step,
        "collective_ip": sim.collective_ip,
        "collective_du": sim.collective_du,
        "knowledge_board": sim.knowledge_board.to_dict(),
        "world_map": sim.world_map.to_dict(),
        "agents": [
            {
                "agent_id": agent.agent_id,
                "ip": agent.state.ip,
                "du": agent.state.du,
                "mood": agent.state.mood_level,
            }
            for agent in sim.agents
        ],
        "seed": sim.seed,
        "rng_state": capture_rng_state(),
        "trace_hash": sim._last_trace_hash,
    }
    snapshot_no_vector = {
        **{k: v for k, v in snapshot.items() if k != "trace_hash"},
        "knowledge_board": {
            k: v for k, v in snapshot["knowledge_board"].items() if k != "vector"
        },
        "world_map": {k: v for k, v in snapshot["world_map"].items() if k != "vector"},
    }
    snapshot["trace_hash"] = compute_trace_hash(snapshot_no_vector)
    save_snapshot(sim.current_step, snapshot, directory=directory)
    return directory / f"snapshot_{sim.current_step}.json"


def _summarize_targets(
    metrics: dict[str, list[tuple[int, float]]],
    evaluation_targets: dict[str, dict[str, float | int | str]] | None,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    if not evaluation_targets:
        return summary

    for metric, target in evaluation_targets.items():
        if not isinstance(target, dict):
            continue
        values = [float(value) for _, value in metrics.get(metric, [])]
        if not values:
            summary[metric] = {
                "status": "missing",
                "reason": "No samples recorded for this metric.",
            }
            continue

        checks: dict[str, dict[str, object]] = {}
        status: str = "pass"
        for field, raw_threshold in target.items():
            if not isinstance(raw_threshold, (int, float)):
                checks[field] = {
                    "threshold": raw_threshold,
                    "passed": None,
                    "reason": "Non-numeric threshold is not evaluated.",
                }
                if status == "pass":
                    status = "unknown"
                continue

            threshold = float(raw_threshold)
            observed: float
            passed: bool | None

            if field == "max_count":
                observed = max(values)
                passed = observed <= threshold
            elif field == "max_value":
                observed = max(values)
                passed = observed <= threshold
            elif field == "min_value":
                observed = min(values)
                passed = observed >= threshold
            elif field == "max_delta":
                deltas = [abs(curr - prev) for prev, curr in zip(values[:-1], values[1:])]
                observed = max(deltas) if deltas else 0.0
                passed = observed <= threshold
            elif field == "max_variance":
                observed = statistics.pvariance(values) if len(values) > 1 else 0.0
                passed = observed <= threshold
            else:
                checks[field] = {
                    "threshold": threshold,
                    "passed": None,
                    "reason": "Unsupported target field.",
                }
                if status == "pass":
                    status = "unknown"
                continue

            checks[field] = {
                "threshold": threshold,
                "observed": observed,
                "passed": passed,
            }
            if not passed:
                status = "fail"

        if not checks:
            summary[metric] = {
                "status": "unknown",
                "reason": "No recognized target fields for evaluation.",
            }
            continue

        summary[metric] = {"status": status, "checks": checks}

    return summary


def _write_readme(
    entries: Iterable[tuple[str, Path]],
    target_summary: dict[str, dict[str, object]] | None = None,
) -> None:
    lines = [
        "# Signature Demo Results",
        "",
        "Artifacts generated by `scripts/run_signature_demo.py`:",
        "",
    ]
    for label, path in entries:
        try:
            rel = path.relative_to(RESULT_DIR)
        except ValueError:
            rel = path
        suffix = "/" if path.is_dir() else ""
        lines.append(f"- **{label}**: `{rel}{suffix}`")
    if target_summary:
        lines.extend(["", "## Evaluation Target Summary", ""])
        for metric, details in target_summary.items():
            status = str(details.get("status", "unknown")).upper()
            lines.append(f"- **{metric}**: {status}")
            checks = details.get("checks")
            if isinstance(checks, dict):
                for field, check_details in checks.items():
                    if not isinstance(check_details, dict):
                        continue
                    observed = check_details.get("observed")
                    threshold = check_details.get("threshold")
                    passed = check_details.get("passed")
                    note_parts: list[str] = []
                    if isinstance(observed, (int, float)):
                        note_parts.append(f"observed={observed:.3f}")
                    if isinstance(threshold, (int, float)):
                        note_parts.append(f"target={threshold:.3f}")
                    if isinstance(passed, bool):
                        note_parts.append("pass" if passed else "fail")
                    reason = check_details.get("reason")
                    if isinstance(reason, str):
                        note_parts.append(reason)
                    summary_line = ", ".join(note_parts) if note_parts else "no details"
                    lines.append(f"  - {field}: {summary_line}")
            reason = details.get("reason")
            if isinstance(reason, str):
                lines.append(f"  - Note: {reason}")

    README_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    """Execute the signature_demo scenario and save outputs."""
    _clean_results_dir()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    event_log_path = RESULT_DIR / "event_log.jsonl"
    os.environ["EVENT_LOG_PATH"] = str(event_log_path)

    (
        scenario_desc,
        steps_override,
        agents_override,
        beats,
        evaluation_hooks,
        evaluation_targets,
        success_metrics,
    ) = load_scenario(str(SCENARIO_PATH))
    steps = steps_override or 0
    num_agents = agents_override or 3

    sim = create_simulation(
        num_agents=num_agents,
        steps=steps,
        scenario=scenario_desc,
        beats=beats,
        seed=42,
        evaluation_hook_names=evaluation_hooks,
        evaluation_targets=evaluation_targets,
        success_metrics=success_metrics,
    )

    await sim.async_run(steps)

    metrics_path = RESULT_DIR / "metrics.json"
    metrics = export_traces.load_metrics(event_log_path)
    target_summary = _summarize_targets(
        metrics, data.get("evaluation_targets") if isinstance(data, dict) else None
    )
    metrics_output: dict[str, object] = {key: value for key, value in metrics.items()}
    if target_summary:
        metrics_output["_target_summary"] = target_summary
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_output, fh, indent=2)

    snapshots_dir = RESULT_DIR / "snapshots"
    snapshot_path = _create_snapshot(sim, snapshots_dir)
    replay_path = event_log.store_replay_slice(0, sim.current_step, directory=snapshots_dir)

    traces_path = RESULT_DIR / "traces.jsonl"
    bundle_path = RESULT_DIR / "signature_demo_bundle.zip"
    export_args = [
        "--events",
        str(event_log_path),
        "--output",
        str(traces_path),
        "--bundle",
        str(bundle_path),
        "--snapshots-dir",
        str(snapshots_dir),
    ]
    if export_traces.main(export_args) != 0:
        raise RuntimeError("Failed to export traces from event log")

    plots_dir = RESULT_DIR / "plots"
    plot_paths = _generate_plots(metrics, plots_dir)

    artifact_entries: list[tuple[str, Path]] = [
        ("Event log", event_log_path),
        ("Metrics", metrics_path),
        ("Trace dataset", traces_path),
        ("Replay slice", replay_path),
        ("Snapshot", snapshot_path),
        ("Bundle archive", bundle_path),
    ]
    for plot_path in sorted(plot_paths):
        artifact_entries.append((f"Plot: {plot_path.stem}", plot_path))
    _write_readme(artifact_entries, target_summary)

    print(f"Event log: {event_log_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Traces: {traces_path}")
    print(f"Replay slice: {replay_path}")
    print(f"Bundle: {bundle_path}")
    if plot_paths:
        print(f"Plots directory: {plots_dir}")


if __name__ == "__main__":
    asyncio.run(main())
