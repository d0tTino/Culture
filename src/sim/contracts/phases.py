from __future__ import annotations

from dataclasses import dataclass

KERNEL_PHASE_SEQUENCE: tuple[str, ...] = ("ingest", "decide", "apply", "persist", "publish")


@dataclass(frozen=True, slots=True)
class KernelPhaseContract:
    version: str = "1.0.0"
    sequence: tuple[str, ...] = KERNEL_PHASE_SEQUENCE


KERNEL_PHASE_CONTRACT = KernelPhaseContract()
