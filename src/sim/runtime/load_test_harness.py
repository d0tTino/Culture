from __future__ import annotations

import asyncio
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

from src.sim.analytics import compute_user_value_kpis
from src.sim.runtime.actor_runtime import RuntimeOrchestrator, replay_event_log


@dataclass(frozen=True, slots=True)
class LoadScenarioResult:
    agent_count: int
    steps: int
    throughput_events_per_sec: float
    p50_step_latency_ms: float
    p95_step_latency_ms: float
    memory_growth_bytes: int
    deterministic_replay: bool


async def run_synthetic_scenario(agent_count: int, *, steps: int = 10) -> LoadScenarioResult:
    runtime = RuntimeOrchestrator(agent_count=agent_count)
    step_latencies_ms: list[float] = []

    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()

    started = time.perf_counter()
    for step in range(steps):
        step_started = time.perf_counter()
        for agent_id in runtime.agent_ids:
            runtime.submit_agent_step(agent_id=agent_id, step=step)
        await runtime.drain()
        step_elapsed = (time.perf_counter() - step_started) * 1000.0
        step_latencies_ms.append(step_elapsed)

    elapsed = max(time.perf_counter() - started, 1e-9)
    throughput = len(runtime.event_log) / elapsed
    replay_state = replay_event_log(agent_count, runtime.event_log)
    deterministic = replay_state == runtime.state_snapshot()

    after_current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    p50 = statistics.median(step_latencies_ms) if step_latencies_ms else 0.0
    p95 = statistics.quantiles(step_latencies_ms, n=20)[18] if len(step_latencies_ms) > 1 else 0.0
    return LoadScenarioResult(
        agent_count=agent_count,
        steps=steps,
        throughput_events_per_sec=throughput,
        p50_step_latency_ms=p50,
        p95_step_latency_ms=p95,
        memory_growth_bytes=after_current - before_current,
        deterministic_replay=deterministic,
    )


async def run_default_load_suite() -> list[LoadScenarioResult]:
    results: list[LoadScenarioResult] = []
    for agents in (20, 50, 100):
        results.append(await run_synthetic_scenario(agents, steps=10))
    return results


def observability_payload(results: list[LoadScenarioResult]) -> list[dict[str, float | int | bool]]:
    """Render p50/p95 tick latency and memory growth for external observability sinks."""

    return [
        {
            "agent_count": result.agent_count,
            "steps": result.steps,
            "throughput_events_per_sec": result.throughput_events_per_sec,
            "tick_latency_p50_ms": result.p50_step_latency_ms,
            "tick_latency_p95_ms": result.p95_step_latency_ms,
            "memory_growth_bytes": result.memory_growth_bytes,
            "deterministic_replay": result.deterministic_replay,
        }
        for result in results
    ]


@dataclass(frozen=True, slots=True)
class ScenarioABResult:
    scenario_name: str
    variant_a: str
    variant_b: str
    engagement_metrics_a: dict[str, float | int]
    engagement_metrics_b: dict[str, float | int]
    deltas: dict[str, float]


def _to_sim_events(runtime: RuntimeOrchestrator, *, include_interactions: bool) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for envelope in runtime.event_log:
        payload = dict(envelope.payload)
        record: dict[str, Any] = {
            "type": envelope.event_type,
            "step": envelope.step + 1,
            "agent_id": payload.get("agent_id", envelope.from_actor),
        }
        if envelope.event_type == "agent_step":
            record["action_intent"] = "collaborate" if include_interactions else "idle"
            if include_interactions and runtime.agent_ids:
                idx = runtime.agent_ids.index(str(payload.get("agent_id", runtime.agent_ids[0])))
                record["target_agent_id"] = runtime.agent_ids[(idx + 1) % len(runtime.agent_ids)]
        events.append(record)
    return events


def run_scenario_ab_harness(
    *, scenario_name: str, agent_count: int, steps: int = 10
) -> ScenarioABResult:
    """Run an A/B scenario harness and compare engagement-oriented KPI impact."""

    runtime_a = RuntimeOrchestrator(agent_count=agent_count)
    runtime_b = RuntimeOrchestrator(agent_count=agent_count)
    for step in range(steps):
        for agent_id in runtime_a.agent_ids:
            runtime_a.submit_agent_step(agent_id=agent_id, step=step)
        for agent_id in runtime_b.agent_ids:
            runtime_b.submit_agent_step(agent_id=agent_id, step=step)
        asyncio.run(runtime_a.drain())
        asyncio.run(runtime_b.drain())

    report_a = compute_user_value_kpis(events=_to_sim_events(runtime_a, include_interactions=False), knowledge_entries=[])
    report_b = compute_user_value_kpis(events=_to_sim_events(runtime_b, include_interactions=True), knowledge_entries=[])

    metrics_a: dict[str, float | int] = {
        "narrative_continuity_score": report_a.narrative_continuity_score,
        "cross_agent_interaction_diversity": report_a.cross_agent_interaction_diversity,
        "user_intervention_rate": report_a.user_intervention_rate,
        "novelty_score": report_a.novelty_score,
    }
    metrics_b: dict[str, float | int] = {
        "narrative_continuity_score": report_b.narrative_continuity_score,
        "cross_agent_interaction_diversity": report_b.cross_agent_interaction_diversity,
        "user_intervention_rate": report_b.user_intervention_rate,
        "novelty_score": report_b.novelty_score,
    }
    deltas = {
        key: float(metrics_b[key]) - float(metrics_a[key])
        for key in metrics_a
    }
    return ScenarioABResult(
        scenario_name=scenario_name,
        variant_a="baseline",
        variant_b="interaction_enhanced",
        engagement_metrics_a=metrics_a,
        engagement_metrics_b=metrics_b,
        deltas=deltas,
    )
