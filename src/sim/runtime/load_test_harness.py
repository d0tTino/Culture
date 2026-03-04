from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from src.sim.runtime.actor_runtime import RuntimeOrchestrator, replay_event_log


@dataclass(frozen=True, slots=True)
class LoadScenarioResult:
    agent_count: int
    steps: int
    throughput_events_per_sec: float
    p95_step_latency_ms: float
    deterministic_replay: bool


async def run_synthetic_scenario(agent_count: int, *, steps: int = 10) -> LoadScenarioResult:
    runtime = RuntimeOrchestrator(agent_count=agent_count)
    step_latencies_ms: list[float] = []

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

    p95 = statistics.quantiles(step_latencies_ms, n=20)[18] if len(step_latencies_ms) > 1 else 0.0
    return LoadScenarioResult(
        agent_count=agent_count,
        steps=steps,
        throughput_events_per_sec=throughput,
        p95_step_latency_ms=p95,
        deterministic_replay=deterministic,
    )


async def run_default_load_suite() -> list[LoadScenarioResult]:
    results: list[LoadScenarioResult] = []
    for agents in (20, 50, 100):
        results.append(await run_synthetic_scenario(agents, steps=10))
    return results
