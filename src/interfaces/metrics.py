"""Prometheus metrics for Culture simulation."""

from typing import Any, cast

# Skip self argument annotation warnings in helper classes
# ruff: noqa: ANN101

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
LLM_CALLS_TOTAL = Counter("llm_calls_total", "Total number of LLM calls")
LLM_ERRORS_TOTAL = Counter("llm_errors_total", "Total number of failed LLM calls")
KNOWLEDGE_BOARD_SIZE = Gauge(
    "knowledge_board_size", "Number of entries currently on the Knowledge Board"
)

# Gas price metrics updated by ``Ledger.calculate_gas_price``
GAS_PRICE_PER_CALL = Gauge("gas_price_per_call", "Current gas price charged per LLM call")
GAS_PRICE_PER_TOKEN = Gauge("gas_price_per_token", "Current gas price charged per generated token")

# Start the metrics HTTP server when this module is imported
try:
    start_http_server(8000)
except Exception:  # pragma: no cover - best effort if port is in use
    pass


def get_llm_latency() -> float:
    """Return the last recorded LLM latency in milliseconds."""
    return float(LLM_LATENCY_MS._value.get())


def get_kb_size() -> int:
    """Return the current Knowledge Board size."""
    return int(KNOWLEDGE_BOARD_SIZE._value.get())


def get_gas_price_per_call() -> float:
    """Return the latest gas price charged per LLM call."""
    return float(GAS_PRICE_PER_CALL._value.get())


def get_gas_price_per_token() -> float:
    """Return the latest gas price charged per generated token."""
    return float(GAS_PRICE_PER_TOKEN._value.get())
