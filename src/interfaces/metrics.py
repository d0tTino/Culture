"""Prometheus metrics for Culture simulation."""

from typing import Any, cast

# Skip self argument annotation warnings in helper classes

try:
    from prometheus_client import Counter, Gauge, start_http_server
except Exception:  # pragma: no cover - optional dependency

    class _Value:
        def __init__(self) -> None:
            self._val = 0

        def get(self) -> int:
            return self._val

    class _Dummy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._value = _Value()

        def __call__(self, *args: object, **kwargs: object) -> "_Dummy":
            return self

        def inc(
            self, amount: int = 1, *args: object, **kwargs: object
        ) -> None:  # pragma: no cover - noop
            self._value._val += amount

        def set(
            self, value: int = 0, *args: object, **kwargs: object
        ) -> None:  # pragma: no cover - noop
            self._value._val = value

    Counter = cast(Any, _Dummy)
    Gauge = cast(Any, _Dummy)

    def start_http_server(*args: object, **kwargs: object) -> None:  # pragma: no cover - noop
        return


# Expose metrics for LLM calls and knowledge board state
LLM_LATENCY_MS = Gauge("llm_latency_ms", "Latency of last LLM call in milliseconds")
LLM_LATENCY_P95_MS = Gauge(
    "llm_latency_p95_ms", "95th percentile latency of recent LLM calls in milliseconds"
)
LLM_CALLS_TOTAL = Counter("llm_calls_total", "Total number of LLM calls")
LLM_ERRORS_TOTAL = Counter("llm_errors_total", "Total number of failed LLM calls")
KNOWLEDGE_BOARD_SIZE = Gauge(
    "knowledge_board_size", "Number of entries currently on the Knowledge Board"
)

# Memory retrieval metrics
MEMORY_RETRIEVALS_TOTAL = Counter(
    "memory_retrievals_total",
    "Total number of successful memory retrievals",
)
MEMORY_RETRIEVAL_ERRORS_TOTAL = Counter(
    "memory_retrieval_errors_total",
    "Total number of failed memory retrievals",
)

# Retrieval Augmented Generation (RAG) metrics
RAG_HIT_RATE = Gauge("rag_hit_rate", "Hit rate for RAG memory retrieval")

# Human interaction metrics
HUMAN_MESSAGES_TOTAL = Counter(
    "human_messages_total",
    "Total number of human messages received",
)

# Gas price metrics updated by ``Ledger.calculate_gas_price``
GAS_PRICE_PER_CALL = Gauge("gas_price_per_call", "Current gas price charged per LLM call")
GAS_PRICE_PER_TOKEN = Gauge("gas_price_per_token", "Current gas price charged per generated token")
LLM_DU_PER_1K_TOKENS = Gauge("llm_du_per_1k_tokens", "DU cost per 1k tokens for the last LLM call")

# Start the metrics HTTP server when this module is imported
try:
    start_http_server(8000)
except Exception:  # pragma: no cover - best effort if port is in use
    pass


def get_llm_latency() -> float:
    """Return the last recorded LLM latency in milliseconds."""
    return float(LLM_LATENCY_MS._value.get())


def get_llm_latency_p95() -> float:
    """Return the 95th percentile LLM latency in milliseconds."""
    return float(LLM_LATENCY_P95_MS._value.get())


def get_kb_size() -> int:
    """Return the current Knowledge Board size."""
    return int(KNOWLEDGE_BOARD_SIZE._value.get())


def get_gas_price_per_call() -> float:
    """Return the latest gas price charged per LLM call."""
    return float(GAS_PRICE_PER_CALL._value.get())


def get_gas_price_per_token() -> float:
    """Return the latest gas price charged per generated token."""
    return float(GAS_PRICE_PER_TOKEN._value.get())


def get_du_per_1k_tokens() -> float:
    """Return the DU cost per 1k tokens for the last call."""
    return float(LLM_DU_PER_1K_TOKENS._value.get())


def get_memory_retrievals() -> int:
    """Return the total successful memory retrievals."""
    return int(MEMORY_RETRIEVALS_TOTAL._value.get())


def get_memory_retrieval_errors() -> int:
    """Return the total failed memory retrievals."""
    return int(MEMORY_RETRIEVAL_ERRORS_TOTAL._value.get())


def get_rag_hit_rate() -> float:
    """Return the current RAG hit rate."""
    return float(RAG_HIT_RATE._value.get())


def get_human_messages() -> int:
    """Return the total human messages received."""
    return int(HUMAN_MESSAGES_TOTAL._value.get())


__all__ = [
    "GAS_PRICE_PER_CALL",
    "GAS_PRICE_PER_TOKEN",
    "HUMAN_MESSAGES_TOTAL",
    "KNOWLEDGE_BOARD_SIZE",
    "LLM_CALLS_TOTAL",
    "LLM_DU_PER_1K_TOKENS",
    "LLM_ERRORS_TOTAL",
    "LLM_LATENCY_MS",
    "LLM_LATENCY_P95_MS",
    "MEMORY_RETRIEVALS_TOTAL",
    "MEMORY_RETRIEVAL_ERRORS_TOTAL",
    "RAG_HIT_RATE",
    "Counter",
    "Gauge",
    "get_du_per_1k_tokens",
    "get_gas_price_per_call",
    "get_gas_price_per_token",
    "get_human_messages",
    "get_kb_size",
    "get_llm_latency",
    "get_llm_latency_p95",
    "get_memory_retrieval_errors",
    "get_memory_retrievals",
    "get_rag_hit_rate",
    "start_http_server",
]
