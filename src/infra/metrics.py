"""Internal metrics tracking for simulation infrastructure."""

from __future__ import annotations

from collections import deque
from typing import Deque

from src.interfaces import metrics as prom_metrics

# Store the most recent DU-per-1k-tokens value
_last_du_per_1k_tokens: float = 0.0

# Keep a rolling window of recent LLM latencies in milliseconds
_LATENCY_SAMPLES: Deque[float] = deque(maxlen=100)


def record_du_per_1k_tokens(value: float) -> None:
    """Record DU cost per 1k tokens for the latest LLM call."""
    global _last_du_per_1k_tokens
    _last_du_per_1k_tokens = float(value)
    prom_metrics.LLM_DU_PER_1K_TOKENS.set(_last_du_per_1k_tokens)


def get_du_per_1k_tokens() -> float:
    """Return the most recently recorded DU-per-1k-tokens value."""
    return _last_du_per_1k_tokens


def record_llm_latency(latency_ms: float) -> None:
    """Record latency for an LLM call and update p95 statistics."""
    prom_metrics.LLM_LATENCY_MS.set(latency_ms)
    _LATENCY_SAMPLES.append(latency_ms)
    if _LATENCY_SAMPLES:
        ordered = sorted(_LATENCY_SAMPLES)
        idx = int(0.95 * (len(ordered) - 1))
        prom_metrics.LLM_LATENCY_P95_MS.set(ordered[idx])


def get_llm_latency_p95() -> float:
    """Return the p95 latency of recent LLM calls in milliseconds."""
    if not _LATENCY_SAMPLES:
        return 0.0
    ordered = sorted(_LATENCY_SAMPLES)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


__all__ = [
    "get_du_per_1k_tokens",
    "get_llm_latency_p95",
    "record_du_per_1k_tokens",
    "record_llm_latency",
]
