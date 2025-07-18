"""Unified memory service for agent operations."""

from __future__ import annotations

from typing import Any

from typing_extensions import Self

from .multi_layer_retriever import MultiLayerRetriever
from .semantic_memory_manager import SemanticMemoryManager
from .vector_store import ChromaVectorStoreManager


class MemoryService:
    """Wrap vector store and semantic memory managers."""

    def __init__(
        self: Self,
        vector_store: ChromaVectorStoreManager | None = None,
        semantic_manager: SemanticMemoryManager | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.semantic_manager = semantic_manager
        self.retriever = MultiLayerRetriever(vector_store, semantic_manager)

    def add_memory(
        self: Self,
        agent_id: str,
        step: int,
        event_type: str,
        content: str,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self.vector_store:
            return ""
        return self.vector_store.add_memory(
            agent_id, step, event_type, content, memory_type, metadata
        )

    async def retrieve_relevant_memories(
        self: Self, agent_id: str, query: str = "", k: int = 5
    ) -> list[dict[str, Any]]:
        return await self.retriever.retrieve(agent_id, query, k)

    def retrieve_semantic_context(
        self: Self, agent_id: str, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        if not self.semantic_manager:
            return []
        return self.semantic_manager.retrieve_context(agent_id, query, k)

    def get_recent_semantic_summaries(self: Self, agent_id: str, limit: int = 3) -> list[str]:
        return self.retriever.get_recent_semantic_summaries(agent_id, limit)

    async def retrieve_episodic_and_update_semantic(
        self: Self, agent_id: str, query: str = "", k: int = 5
    ) -> list[dict[str, Any]]:
        return await self.retriever.retrieve_and_update_semantic(agent_id, query, k)

    def blend_with_recent_semantic(
        self: Self, agent_id: str, episodic_summary: str, limit: int = 3
    ) -> str:
        """Blend an episodic summary with recent semantic summaries."""
        return self.retriever.blend_with_recent_semantic(agent_id, episodic_summary, limit)

    async def get_context_pipeline(
        self: Self,
        agent_id: str,
        query: str = "",
        k: int = 5,
        semantic_limit: int = 3,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Full retrieval pipeline returning episodic memories and semantic summaries."""
        episodic = await self.retrieve_episodic_and_update_semantic(agent_id, query, k)
        semantic = self.get_recent_semantic_summaries(agent_id, semantic_limit)
        return episodic, semantic

    async def run_semantic_job(
        self: Self,
        agent_id: str,
        episodic_memories: list[dict[str, Any]] | None = None,
    ) -> None:
        await self.retriever.run_semantic_job(agent_id, episodic_memories)
        return None

    def consolidate_daily_memories(
        self: Self, agent_id: str, start_step: int, end_step: int
    ) -> None:
        if not self.vector_store:
            return None
        self.vector_store.consolidate_daily_memories(agent_id, start_step, end_step)
        return None

    async def aconsolidate_daily_memories(
        self: Self, agent_id: str, start_step: int, end_step: int
    ) -> None:
        if not self.vector_store:
            return None
        await self.vector_store.aconsolidate_daily_memories(agent_id, start_step, end_step)
        return None

    def prune_expired(self: Self, ttl_seconds: int) -> int:
        if not self.vector_store:
            return 0
        return self.vector_store.prune(ttl_seconds)

    def prune_mus(
        self: Self,
        l1_threshold: float = 0.2,
        l2_threshold: float = 0.3,
        l2_age_days: int = 30,
        l1_min_age_days: int = 0,
        l2_min_age_days: int = 0,
    ) -> int:
        if not self.vector_store:
            return 0
        return self.vector_store.prune_memories_hybrid(
            l1_threshold,
            l2_threshold,
            l2_age_days,
            l1_min_age_days,
            l2_min_age_days,
        )

    def close(self: Self) -> None:
        if self.vector_store and hasattr(self.vector_store, "close"):
            try:
                self.vector_store.close()
            except Exception:
                pass
        return None
