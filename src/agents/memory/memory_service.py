"""Unified memory service for agent operations."""

from __future__ import annotations

from typing import Any

from typing_extensions import Self

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
        episodic: list[dict[str, Any]] = []
        if self.vector_store:
            episodic = await self.vector_store.aretrieve_relevant_memories(agent_id, query, k)

        semantic: list[dict[str, Any]] = []
        if self.semantic_manager and self.vector_store:
            import asyncio

            import numpy as np

            semantic = await asyncio.to_thread(
                self.semantic_manager.retrieve_context, agent_id, query, k
            )
            q_emb = np.array(self.vector_store.get_embedding(query), dtype=float)
            for mem in semantic:
                emb = np.array(
                    self.vector_store.get_embedding(mem.get("content", "")), dtype=float
                )
                score = float(emb @ q_emb / (np.linalg.norm(emb) * np.linalg.norm(q_emb) + 1e-8))
                mem["relevance_score"] = score

        combined = episodic + semantic
        combined.sort(key=lambda m: m.get("relevance_score", 0.0), reverse=True)
        return combined[:k]

    def retrieve_semantic_context(
        self: Self, agent_id: str, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        if not self.semantic_manager:
            return []
        return self.semantic_manager.retrieve_context(agent_id, query, k)

    def get_recent_semantic_summaries(self: Self, agent_id: str, limit: int = 3) -> list[str]:
        if not self.semantic_manager:
            return []
        return self.semantic_manager.get_recent_summaries(agent_id, limit)

    async def run_semantic_job(
        self: Self,
        agent_id: str,
        episodic_memories: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.semantic_manager:
            return None
        await self.semantic_manager.run_nightly_job(agent_id, episodic_memories)
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
