from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StepContext:
    """Per-step runtime state shared across phase components."""

    max_turns: int
    queue_depth: int = 0
    phase_order: list[str] = field(default_factory=list)
    planned_outputs: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
