"""Multi-layer retrieval combining episodic and semantic memories."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import tiktoken
from opentelemetry import trace
from typing_extensions import Self

from src.infra import metrics as infra_metrics
from src.interfaces import metrics

from .semantic_memory_manager import SemanticMemoryManager
from .vector_store import ChromaVectorStoreManager


def _extract_recall_score(memories: Iterable[dict[str, Any]]) -> float | None:
    """Return the first recall@5-style score found in the provided memories."""

    recall_keys = ("recall_p5", "recall@5", "p_at_5", "recall")
    for memory in memories:
        if not isinstance(memory, dict):
            continue

        for key in recall_keys:
            value = memory.get(key)
            if isinstance(value, (int, float)):
                return float(value)

        metrics_payload = memory.get("metrics")
        if isinstance(metrics_payload, dict):
            for key in recall_keys:
                value = metrics_payload.get(key)
                if isinstance(value, (int, float)):
                    return float(value)

    return None

tracer = trace.get_tracer(__name__)


class MultiLayerRetriever:
    """Retrieve from both episodic and semantic layers."""

    def __init__(
        self: Self,
        vector_store: ChromaVectorStoreManager | None = None,
        semantic_manager: SemanticMemoryManager | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.semantic_manager = semantic_manager
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")

    async def retrieve(
        self: Self,
        agent_id: str,
        query: str = "",
        k: int = 5,
        token_budget: int | None = None,
    ) -> list[dict[str, Any]]:
        with tracer.start_as_current_span("memory.retrieve") as span:
            span.set_attribute("memory.agent_id", agent_id)
            span.set_attribute("memory.query.length", len(query))
            span.set_attribute("memory.k", k)
            start_total = time.perf_counter()
            recall_score: float | None = None
            try:
                episodic: list[dict[str, Any]] = []
                if self.vector_store:
                    with tracer.start_as_current_span("memory.episodic_retrieve") as e_span:
                        e_span.set_attribute("memory.path", "episodic")
                        e_span.set_attribute("llm.tokens.prompt", 0)
                        e_span.set_attribute("llm.tokens.completion", 0)
                        e_span.set_attribute("llm.tokens.total", 0)
                        start = time.perf_counter()
                        try:
                            episodic = await self.vector_store.aretrieve_relevant_memories(
                                agent_id, query, k
                            )
                        finally:
                            e_span.set_attribute(
                                "memory.latency_ms",
                                (time.perf_counter() - start) * 1000,
                            )

                semantic: list[dict[str, Any]] = []
                if self.semantic_manager:
                    import asyncio

                    with tracer.start_as_current_span("memory.semantic_retrieve") as s_span:
                        s_span.set_attribute("memory.path", "semantic")
                        s_span.set_attribute("llm.tokens.prompt", 0)
                        s_span.set_attribute("llm.tokens.completion", 0)
                        s_span.set_attribute("llm.tokens.total", 0)
                        start = time.perf_counter()
                        try:
                            semantic = await asyncio.to_thread(
                                self.semantic_manager.retrieve_context_with_scores,
                                agent_id,
                                query,
                                k,
                            )
                        finally:
                            s_span.set_attribute(
                                "memory.latency_ms",
                                (time.perf_counter() - start) * 1000,
                            )

                combined: list[dict[str, Any]] = []
                if token_budget is not None:
                    e_idx = s_idx = 0
                    tokens = 0
                    while tokens < token_budget and (
                        e_idx < len(episodic) or s_idx < len(semantic)
                    ):
                        e_mem = episodic[e_idx] if e_idx < len(episodic) else None
                        s_mem = semantic[s_idx] if s_idx < len(semantic) else None
                        choose_semantic = False
                        if s_mem is not None and (
                            e_mem is None
                            or s_mem.get("relevance_score", 0.0)
                            > e_mem.get("relevance_score", 0.0)
                        ):
                            choose_semantic = True
                        mem = s_mem if choose_semantic else e_mem
                        if mem is None:
                            break
                        text = str(mem.get("content", ""))
                        mem_tokens = len(self.tokenizer.encode(text))
                        if tokens + mem_tokens > token_budget:
                            break
                        combined.append(mem)
                        tokens += mem_tokens
                        if choose_semantic:
                            s_idx += 1
                        else:
                            e_idx += 1
                else:
                    combined = episodic + semantic

                combined.sort(key=lambda m: m.get("relevance_score", 0.0), reverse=True)
                metrics.MEMORY_RETRIEVALS_TOTAL.inc()
                span.set_attribute("memory.results", len(combined))
                top_k = combined[:k]
                recall_score = _extract_recall_score(top_k)
                if recall_score is not None:
                    span.set_attribute("memory.recall_p5", recall_score)
                return top_k
            except Exception:
                metrics.MEMORY_RETRIEVAL_ERRORS_TOTAL.inc()
                raise
            finally:
                latency_ms = (time.perf_counter() - start_total) * 1000
                span.set_attribute("memory.latency_ms", latency_ms)
                infra_metrics.record_retrieval_latency(latency_ms)
                record_recall = getattr(infra_metrics, "record_recall_p5", None)
                if callable(record_recall) and recall_score is not None:
                    try:
                        record_recall(recall_score)
                    except Exception:  # pragma: no cover - defensive
                        pass

    async def retrieve_and_update_semantic(
        self: Self, agent_id: str, query: str = "", k: int = 5
    ) -> list[dict[str, Any]]:
        with tracer.start_as_current_span("memory.retrieve_and_update_semantic") as span:
            span.set_attribute("memory.agent_id", agent_id)
            span.set_attribute("memory.query.length", len(query))
            span.set_attribute("memory.k", k)
            start_total = time.perf_counter()
            recall_score: float | None = None
            try:
                episodic = []
                if self.vector_store:
                    with tracer.start_as_current_span("memory.episodic_retrieve") as e_span:
                        e_span.set_attribute("memory.path", "episodic")
                        e_span.set_attribute("llm.tokens.prompt", 0)
                        e_span.set_attribute("llm.tokens.completion", 0)
                        e_span.set_attribute("llm.tokens.total", 0)
                        start = time.perf_counter()
                        try:
                            episodic = await self.vector_store.aretrieve_relevant_memories(
                                agent_id, query, k
                            )
                        finally:
                            e_span.set_attribute(
                                "memory.latency_ms",
                                (time.perf_counter() - start) * 1000,
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
                span.set_attribute("memory.results", len(episodic))
                recall_score = _extract_recall_score(episodic)
                if recall_score is not None:
                    span.set_attribute("memory.recall_p5", recall_score)
                return episodic
            except Exception:
                metrics.MEMORY_RETRIEVAL_ERRORS_TOTAL.inc()
                raise
            finally:
                latency_ms = (time.perf_counter() - start_total) * 1000
                span.set_attribute("memory.latency_ms", latency_ms)
                infra_metrics.record_retrieval_latency(latency_ms)
                record_recall = getattr(infra_metrics, "record_recall_p5", None)
                if callable(record_recall) and recall_score is not None:
                    try:
                        record_recall(recall_score)
                    except Exception:  # pragma: no cover - defensive
                        pass

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
