"""Utilities for evaluating retrieval quality and performance."""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Sequence


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
