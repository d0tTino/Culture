"""Utilities for evaluating retrieval quality and performance."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Sequence
from typing import Any, Callable


def p95(values: Sequence[float]) -> float:
    """Return the 95th percentile of ``values``.

    Values are assumed to be non-empty and expressed in the same units
    (e.g. seconds). If ``values`` is empty, ``0.0`` is returned.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = int(0.95 * (len(sorted_vals) - 1))
    return sorted_vals[index]


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Return precision at ``k`` for retrieved versus relevant IDs."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for _id in top_k if _id in relevant_ids)
    return hits / float(min(k, len(top_k)))


async def time_call(coro: Callable[[], Awaitable[Sequence[Any]]]) -> tuple[Sequence[Any], float]:
    """Execute ``coro`` and return its result and runtime in seconds."""
    start = time.perf_counter()
    result = await coro()
    return result, time.perf_counter() - start
