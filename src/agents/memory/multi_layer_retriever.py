"""Multi-layer retrieval combining episodic and semantic memories."""

from __future__ import annotations

import time
from typing import Any

from opentelemetry import trace
from typing_extensions import Self

from src.interfaces import metrics

from .semantic_memory_manager import SemanticMemoryManager
from .vector_store import ChromaVectorStoreManager

tracer = trace.get_tracer(__name__)


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
        try:
            episodic: list[dict[str, Any]] = []
            if self.vector_store:
                with tracer.start_as_current_span("memory.episodic_retrieve") as span:
                    span.set_attribute("llm.tokens.prompt", 0)
                    span.set_attribute("llm.tokens.completion", 0)
                    span.set_attribute("llm.tokens.total", 0)
                    start = time.perf_counter()
                    try:
                        episodic = await self.vector_store.aretrieve_relevant_memories(
                            agent_id, query, k
                        )
                    finally:
                        span.set_attribute(
                            "memory.latency_ms", (time.perf_counter() - start) * 1000
                        )

            semantic: list[dict[str, Any]] = []
            if self.semantic_manager:
                import asyncio

                with tracer.start_as_current_span("memory.semantic_retrieve") as span:
                    span.set_attribute("llm.tokens.prompt", 0)
                    span.set_attribute("llm.tokens.completion", 0)
                    span.set_attribute("llm.tokens.total", 0)
                    start = time.perf_counter()
                    try:
                        semantic = await asyncio.to_thread(
                            self.semantic_manager.retrieve_context_with_scores,
                            agent_id,
                            query,
                            k,
                        )
                    finally:
                        span.set_attribute(
                            "memory.latency_ms", (time.perf_counter() - start) * 1000
                        )

            combined = episodic + semantic
            combined.sort(key=lambda m: m.get("relevance_score", 0.0), reverse=True)
            metrics.MEMORY_RETRIEVALS_TOTAL.inc()
            return combined[:k]
        except Exception:
            metrics.MEMORY_RETRIEVAL_ERRORS_TOTAL.inc()
            raise

    async def retrieve_and_update_semantic(
        self: Self, agent_id: str, query: str = "", k: int = 5
    ) -> list[dict[str, Any]]:
        try:
            episodic = []
            if self.vector_store:
                with tracer.start_as_current_span("memory.episodic_retrieve") as span:
                    span.set_attribute("llm.tokens.prompt", 0)
                    span.set_attribute("llm.tokens.completion", 0)
                    span.set_attribute("llm.tokens.total", 0)
                    start = time.perf_counter()
                    try:
                        episodic = await self.vector_store.aretrieve_relevant_memories(
                            agent_id, query, k
                        )
                    finally:
                        span.set_attribute(
                            "memory.latency_ms", (time.perf_counter() - start) * 1000
                        )
            if self.semantic_manager:
                try:
                    await self.semantic_manager.run_nightly_job(agent_id, episodic)
                except Exception:  # pragma: no cover - defensive
                    import logging

                    logging.getLogger(__name__).error(
                        "Semantic consolidation failed", exc_info=True
                    )
            metrics.MEMORY_RETRIEVALS_TOTAL.inc()
            return episodic
        except Exception:
            metrics.MEMORY_RETRIEVAL_ERRORS_TOTAL.inc()
            raise

    def get_recent_semantic_summaries(self: Self, agent_id: str, limit: int = 3) -> list[str]:
        if not self.semantic_manager:
            metrics.MEMORY_RETRIEVAL_ERRORS_TOTAL.inc()
            return []
        try:
            result = self.semantic_manager.get_recent_summaries(agent_id, limit)
            metrics.MEMORY_RETRIEVALS_TOTAL.inc()
            return result
        except Exception:
            metrics.MEMORY_RETRIEVAL_ERRORS_TOTAL.inc()
            raise

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
