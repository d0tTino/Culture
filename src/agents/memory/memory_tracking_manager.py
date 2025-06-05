"""Memory usage tracking utilities for agent memories."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from typing_extensions import Self

logger = logging.getLogger(__name__)


class MemoryTrackingManager:
    """Manage retrieval usage statistics for ChromaDB memories."""

    def __init__(self: Self, collection: Any) -> None:
        self.collection = collection

    def update_usage_stats(
        self: Self,
        memory_ids: list[str],
        relevance_scores: list[float] | None = None,
        *,
        increment_count: bool = True,
    ) -> None:
        """Update retrieval metadata for memories."""
        if not memory_ids:
            return

        try:
            results = self.collection.get(ids=memory_ids, include=["metadatas"])
            if not results or not results.get("metadatas"):
                logger.warning("No metadata found for memories: %s", memory_ids)
                return
            current_time = datetime.utcnow().isoformat()
            updated: list[dict[str, Any]] = []
            metadatas = results["metadatas"]
            for i, memory_id in enumerate(memory_ids):
                if metadatas is None or i >= len(metadatas):
                    continue
                metadata = dict(metadatas[i])
                if increment_count:
                    metadata["retrieval_count"] = int(metadata.get("retrieval_count", 0)) + 1
                    if "first_retrieved_at" not in metadata:
                        metadata["first_retrieved_at"] = datetime.now(timezone.utc).isoformat()
                    metadata["last_retrieved_at"] = datetime.now(timezone.utc).isoformat()
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
                updated.append(metadata)
            if updated:
                self.collection.update(ids=memory_ids, metadatas=updated)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Error updating memory usage statistics: %s", exc, exc_info=True)

    def calculate_mus(self: Self, metadata: dict[str, Any]) -> float:
        """Compute the Memory Utility Score for a memory."""
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
                days_since = (now - last_dt).total_seconds() / (24 * 3600)
                if days_since < 0:
                    days_since = 0.01
                recs = 1.0 / (1.0 + days_since)
            except Exception:  # pragma: no cover - defensive
                recs = 0.0
        mus = (0.4 * rfs) + (0.4 * rs) + (0.2 * recs)
        return mus
