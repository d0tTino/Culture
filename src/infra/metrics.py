"""Internal metrics tracking for simulation infrastructure."""

from __future__ import annotations

from collections import deque

from src.interfaces import metrics as prom_metrics

from .ledger import ledger

# Store the most recent DU-per-1k-tokens value
_last_du_per_1k_tokens: float = 0.0
# Track DU-per-1k-tokens per agent for quick lookup
_agent_du_per_1k_tokens: dict[str, float] = {}

# Keep a rolling window of recent LLM latencies in milliseconds
_LATENCY_SAMPLES: deque[float] = deque(maxlen=100)
# Per-agent latency samples for p95 calculations
_LATENCY_SAMPLES_PER_AGENT: dict[str, deque[float]] = {}

# Keep rolling windows for retrieval benchmark metrics
_RETRIEVAL_LATENCY_SAMPLES: deque[float] = deque(maxlen=100)
_RECALL_P5_SAMPLES: deque[float] = deque(maxlen=100)


def record_du_per_1k_tokens(agent_id: str, value: float) -> None:
    """Record DU cost per 1k tokens for the latest LLM call."""
    global _last_du_per_1k_tokens
    _last_du_per_1k_tokens = float(value)
    _agent_du_per_1k_tokens[agent_id] = _last_du_per_1k_tokens
    prom_metrics.LLM_DU_PER_1K_TOKENS.set(_last_du_per_1k_tokens)
    if hasattr(prom_metrics.AGENT_DU_PER_1K_TOKENS, "labels"):
        prom_metrics.AGENT_DU_PER_1K_TOKENS.labels(agent_id=agent_id).set(_last_du_per_1k_tokens)
    try:  # pragma: no cover - optional dependency
        ledger.record_du_per_1k_tokens(agent_id, value)
    except Exception:
        pass


def get_du_per_1k_tokens() -> float:
    """Return the most recently recorded DU-per-1k-tokens value."""
    return _last_du_per_1k_tokens


def get_agent_du_per_1k_tokens(agent_id: str) -> float:
    """Return the DU-per-1k-tokens value for ``agent_id``."""
    return float(_agent_du_per_1k_tokens.get(agent_id, 0.0))


def record_llm_latency(agent_id: str, latency_ms: float) -> None:
    """Record latency for an LLM call and update p95 statistics."""
    prom_metrics.LLM_LATENCY_MS.set(latency_ms)
    _LATENCY_SAMPLES.append(latency_ms)
    if _LATENCY_SAMPLES:
        ordered = sorted(_LATENCY_SAMPLES)
        idx = int(0.95 * (len(ordered) - 1))
        prom_metrics.LLM_LATENCY_P95_MS.set(ordered[idx])

    agent_samples = _LATENCY_SAMPLES_PER_AGENT.setdefault(agent_id, deque(maxlen=100))
    agent_samples.append(latency_ms)
    ordered_agent = sorted(agent_samples)
    idx_agent = int(0.95 * (len(ordered_agent) - 1))
    if hasattr(prom_metrics.AGENT_LLM_LATENCY_P95_MS, "labels"):
        prom_metrics.AGENT_LLM_LATENCY_P95_MS.labels(agent_id=agent_id).set(
            ordered_agent[idx_agent]
        )


def record_retrieval_latency(latency_ms: float) -> None:
    """Record retrieval latency and update p95 statistics."""
    prom_metrics.RETRIEVAL_LATENCY_MS.set(latency_ms)
    _RETRIEVAL_LATENCY_SAMPLES.append(latency_ms)
    if _RETRIEVAL_LATENCY_SAMPLES:
        ordered = sorted(_RETRIEVAL_LATENCY_SAMPLES)
        idx = int(0.95 * (len(ordered) - 1))
        prom_metrics.RETRIEVAL_LATENCY_P95_MS.set(ordered[idx])


def record_recall_p5(value: float) -> None:
    """Record recall@5 and update the running average."""
    _RECALL_P5_SAMPLES.append(float(value))
    if _RECALL_P5_SAMPLES:
        avg = sum(_RECALL_P5_SAMPLES) / len(_RECALL_P5_SAMPLES)
        prom_metrics.RECALL_P5.set(avg)


def get_llm_latency_p95() -> float:
    """Return the p95 latency of recent LLM calls in milliseconds."""
    if not _LATENCY_SAMPLES:
        return 0.0
    ordered = sorted(_LATENCY_SAMPLES)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def get_retrieval_latency_p95() -> float:
    """Return the p95 latency of recent retrievals in milliseconds."""
    if not _RETRIEVAL_LATENCY_SAMPLES:
        return 0.0
    ordered = sorted(_RETRIEVAL_LATENCY_SAMPLES)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def get_recall_p5() -> float:
    """Return the average recall@5 for recent retrievals."""
    if not _RECALL_P5_SAMPLES:
        return 0.0
    return sum(_RECALL_P5_SAMPLES) / len(_RECALL_P5_SAMPLES)


__all__ = [
    "get_du_per_1k_tokens",
    "get_llm_latency_p95",
    "get_recall_p5",
    "get_retrieval_latency_p95",
    "record_du_per_1k_tokens",
    "record_llm_latency",
    "record_recall_p5",
    "record_retrieval_latency",
]
