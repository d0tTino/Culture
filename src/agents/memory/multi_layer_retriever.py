"""Multi-layer retrieval combining episodic and semantic memories."""

from __future__ import annotations

from typing import Any

from typing_extensions import Self

from .semantic_memory_manager import SemanticMemoryManager
from .vector_store import ChromaVectorStoreManager


class MultiLayerRetriever:
    """Retrieve from both episodic and semantic layers."""

    def __init__(
        self: Self,
        vector_store: ChromaVectorStoreManager | None = None,
        semantic_manager: SemanticMemoryManager | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.semantic_manager = semantic_manager

    async def retrieve(
        self: Self, agent_id: str, query: str = "", k: int = 5
    ) -> list[dict[str, Any]]:
        episodic: list[dict[str, Any]] = []
        if self.vector_store:
            episodic = await self.vector_store.aretrieve_relevant_memories(agent_id, query, k)

        semantic: list[dict[str, Any]] = []
        if self.semantic_manager:
            import asyncio

            semantic = await asyncio.to_thread(
                self.semantic_manager.retrieve_context_with_scores, agent_id, query, k
            )

        combined = episodic + semantic
        combined.sort(key=lambda m: m.get("relevance_score", 0.0), reverse=True)
        return combined[:k]

    async def retrieve_and_update_semantic(
        self: Self, agent_id: str, query: str = "", k: int = 5
    ) -> list[dict[str, Any]]:
        episodic = []
        if self.vector_store:
            episodic = await self.vector_store.aretrieve_relevant_memories(agent_id, query, k)
        if self.semantic_manager:
            try:
                await self.semantic_manager.run_nightly_job(agent_id, episodic)
            except Exception:  # pragma: no cover - defensive
                import logging

                logging.getLogger(__name__).error("Semantic consolidation failed", exc_info=True)
        return episodic

    def get_recent_semantic_summaries(self: Self, agent_id: str, limit: int = 3) -> list[str]:
        if not self.semantic_manager:
            return []
        return self.semantic_manager.get_recent_summaries(agent_id, limit)

    def blend_with_recent_semantic(
        self: Self, agent_id: str, episodic_summary: str, limit: int = 3
    ) -> str:
        if not self.semantic_manager:
            return episodic_summary
        return self.semantic_manager.blend_episodic_and_semantic(agent_id, episodic_summary, limit)

    async def run_semantic_job(
        self: Self,
        agent_id: str,
        episodic_memories: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.semantic_manager:
            return None
        await self.semantic_manager.run_nightly_job(agent_id, episodic_memories)
        return None
