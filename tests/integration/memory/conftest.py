import pytest

from src.utils.retrieval_metrics import precision_at_k, time_call


@pytest.fixture
def recall_benchmark():
    """Return helper to measure precision@k and latency for a retrieval coroutine."""

    async def _bench(retrieval_coro, relevant_ids, k):
        results, latency = await time_call(retrieval_coro)
        ids = [m.get("id") for m in results]
        metrics = {
            "p_at_k": precision_at_k(ids, set(relevant_ids), k),
            "latency": latency,
        }
        return metrics

    return _bench
