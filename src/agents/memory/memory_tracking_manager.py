import logging
from collections.abc import Sequence
from typing import Any, Optional

from typing_extensions import Self

from .vector_store import ChromaVectorStoreManager

logger = logging.getLogger(__name__)


class MemoryTrackingManager:
    """Manage usage tracking metadata for agent memories."""

    def __init__(self: Self, vector_store: ChromaVectorStoreManager) -> None:
        self.vector_store = vector_store

    def record_retrieval(
        self: Self, memory_ids: list[str], relevance_scores: Optional[Sequence[float]] = None
    ) -> None:
        """Update usage stats when memories are retrieved."""
        self.vector_store._update_memory_usage_stats(
            memory_ids, list(relevance_scores) if relevance_scores else None, True
        )

    def get_usage_statistics(self: Self, memory_id: str) -> dict[str, Any]:
        """Return usage metadata for a memory."""
        results = self.vector_store.collection.get(ids=[memory_id], include=["metadatas"])
        if results and results.get("metadatas"):
            return dict(results["metadatas"][0])
        return {}

    def calculate_mus(self: Self, memory_id: str) -> float:
        """Calculate Memory Utility Score for a memory."""
        metadata = self.get_usage_statistics(memory_id)
        if not metadata:
            return 0.0
        return self.vector_store._calculate_mus(metadata)
