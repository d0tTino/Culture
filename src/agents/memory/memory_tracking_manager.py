from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .vector_store import ChromaVectorStoreManager

logger = logging.getLogger(__name__)


class MemoryTrackingManager:
    """Manage usage tracking metadata for agent memories."""

    def __init__(self: Self, vector_store: ChromaVectorStoreManager) -> None:
        self.vector_store = vector_store

    def record_retrieval(
        self: Self,
        memory_ids: list[str],
        relevance_scores: Sequence[float] | None = None,
    ) -> None:
        """Update usage stats when memories are retrieved."""
        self.update_usage_stats(memory_ids, relevance_scores, increment_count=True)

    def update_usage_stats(
        self: Self,
        memory_ids: list[str],
        relevance_scores: Sequence[float] | None = None,
        *,
        increment_count: bool = True,
    ) -> None:
        """Update retrieval metadata for memories."""
        if not memory_ids:
            return

        try:
            results = self.vector_store.collection.get(ids=memory_ids, include=["metadatas"])
            if not results or not results.get("metadatas"):
                logger.warning("No metadata found for memories: %s", memory_ids)
                return
            current_time = datetime.utcnow().isoformat()
            updated_metadatas: list[dict[str, Any]] = []
            metadatas = results["metadatas"]
            for i, memory_id in enumerate(memory_ids):
                if metadatas is None or i >= len(metadatas):
                    continue
                metadata = dict(metadatas[i])
                if increment_count:
                    metadata["retrieval_count"] = int(metadata.get("retrieval_count", 0)) + 1
                    if "first_retrieved_at" not in metadata:
                        metadata["first_retrieved_at"] = current_time
                    metadata["last_retrieved_at"] = current_time
                metadata["last_retrieved_timestamp"] = current_time
                if (
                    increment_count
                    and relevance_scores
                    and i < len(relevance_scores)
                    and relevance_scores[i] is not None
                ):
                    score = float(relevance_scores[i])
                    metadata["accumulated_relevance_score"] = (
                        float(metadata.get("accumulated_relevance_score", 0.0)) + score
                    )
                    metadata["retrieval_relevance_count"] = (
                        int(metadata.get("retrieval_relevance_count", 0)) + 1
                    )
                updated_metadatas.append(metadata)
            if updated_metadatas:
                self.vector_store.collection.update(ids=memory_ids, metadatas=updated_metadatas)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Error updating memory usage statistics: %s", exc, exc_info=True)

    def get_usage_statistics(self: Self, memory_id: str) -> dict[str, Any]:
        """Return usage metadata for a memory."""
        results = self.vector_store.collection.get(ids=[memory_id], include=["metadatas"])
        if results and results.get("metadatas"):
            return dict(results["metadatas"][0])
        return {}

    def calculate_mus(self: Self, memory: str | dict[str, Any]) -> float:
        """Calculate Memory Utility Score for a memory."""
        if isinstance(memory, dict):
            metadata = memory
        else:
            metadata = self.get_usage_statistics(memory)
            if not metadata:
                return 0.0

        try:
            retrieval_count = int(metadata.get("retrieval_count", 0))
            accumulated_relevance_score = float(metadata.get("accumulated_relevance_score", 0.0))
            relevance_count = int(metadata.get("retrieval_relevance_count", 0))
            last_retrieved = str(metadata.get("last_retrieved_timestamp", ""))
        except Exception:  # pragma: no cover - defensive
            return 0.0

        rfs = math.log(1 + retrieval_count)
        rs = accumulated_relevance_score / relevance_count if relevance_count > 0 else 0.0
        recs = 0.0
        if last_retrieved:
            try:
                last_dt = datetime.fromisoformat(last_retrieved)
                now = datetime.utcnow()
                days_since = max(0.0, (now - last_dt).total_seconds() / (24 * 3600))
                recs = 1.0 / (1.0 + days_since)
            except Exception:  # pragma: no cover - defensive
                recs = 0.0

        mus = (0.4 * rfs) + (0.4 * rs) + (0.2 * recs)
        return mus
