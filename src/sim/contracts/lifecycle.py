from __future__ import annotations

from dataclasses import dataclass, field

LIFECYCLE_CONTRACT_VERSION = "1.0.0"

LIFECYCLE_PHASES: tuple[str, ...] = (
    "perception",
    "decision",
    "action",
    "post_step",
)


EVENT_STEP_LIFECYCLE_CONTRACTS: dict[str, tuple[str, ...]] = {
    "run_step_order": (
        "start_event_listener must run before kernel dispatch",
        "event_kernel.step is the step execution boundary",
        "evaluation hooks execute after events are produced for a step",
    ),
    "bootstrap_semantics": (
        "when kernel queue is empty, exactly one immediate agent event is seeded",
        "seeded event uses current_agent_index and increments that agent vector clock",
    ),
    "metrics_event_semantics": (
        "evaluation metrics are emitted as SimulationEvent(type='evaluation')",
        "evaluation metrics include the current simulation step",
    ),
}


@dataclass(frozen=True, slots=True)
class LifecycleContract:
    version: str = LIFECYCLE_CONTRACT_VERSION
    phases: tuple[str, ...] = LIFECYCLE_PHASES
    invariants: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(EVENT_STEP_LIFECYCLE_CONTRACTS)
    )


LIFECYCLE_CONTRACT = LifecycleContract()
