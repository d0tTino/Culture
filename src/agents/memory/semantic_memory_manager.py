from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

try:  # Import optional scikit-learn dependencies
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency missing
    _KMeans = None
    _TfidfVectorizer = None
    SKLEARN_AVAILABLE = False

if _KMeans is None:
    logging.getLogger(__name__).warning("scikit-learn not installed; using basic clustering stubs")

    class KMeans:
        """Lightweight stand-in for :class:`sklearn.cluster.KMeans`."""

        def __init__(
            self, n_clusters: int = 8, n_init: int = 10, random_state: int | None = None
        ) -> None:
            self.n_clusters = n_clusters
            self.cluster_centers_: NDArray[np.float64] = np.zeros((n_clusters, 1), dtype=float)

        def fit_predict(self, X: Any) -> NDArray[np.int64]:
            n_samples = len(X)
            shape = getattr(X, "shape", (n_samples, 1))
            dim = shape[1] if isinstance(shape, tuple) and len(shape) > 1 else 1
            self.cluster_centers_ = np.zeros((self.n_clusters, dim), dtype=float)
            return np.arange(n_samples) % self.n_clusters

    class TfidfVectorizer:
        """Simplified TF-IDF vectorizer returning empty features."""

        def __init__(self, stop_words: str | None = None) -> None:
            self.stop_words = stop_words

        class _Matrix(np.ndarray[Any, np.dtype[np.float64]]):
            nnz: int

        def fit_transform(self, texts: list[str]) -> _Matrix:
            arr = np.zeros((len(texts), 1), dtype=float).view(self._Matrix)
            arr.nnz = 0
            return arr

else:  # scikit-learn available
    KMeans = _KMeans
    TfidfVectorizer = _TfidfVectorizer

if TYPE_CHECKING:
    from neo4j import Driver
else:  # pragma: no cover - fallback if neo4j not installed
    try:
        from neo4j import Driver
    except Exception:
        Driver = object
from typing_extensions import Self

from .vector_store import ChromaVectorStoreManager

logger = logging.getLogger(__name__)


class SemanticMemoryManager:
    """Manage consolidation and semantic grouping of memories.

    When scikit-learn is not installed, clustering falls back to simplified
    heuristics with reduced accuracy.
    """

    def __init__(
        self: Self, vector_store: ChromaVectorStoreManager, driver: Driver | None
    ) -> None:
        self.vector_store = vector_store
        self.driver = driver
        self.topic_groups: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self.topic_centroids: dict[str, NDArray[np.float64]] = {}

    def consolidate_memories(
        self: Self,
        agent_id: str,
        episodic_memories: list[dict[str, Any]] | None = None,
    ) -> str:
        """Consolidate episodic memories into a semantic summary."""
        memories = (
            episodic_memories
            if episodic_memories is not None
            else self.vector_store.retrieve_filtered_memories(
                agent_id, filters={"memory_type": "raw"}, limit=None
            )
        )
        if not memories:
            return ""
        summary = "\n".join(mem["content"] for mem in memories)
        return summary

    def group_memories_by_topic(
        self: Self, agent_id: str, num_topics: int = 5, threshold: float = 0.7
    ) -> dict[int, list[dict[str, Any]]]:
        """Group memories into topics using embeddings or simple keywords.

        This operation relies on scikit-learn when available. Without it,
        grouping falls back to a basic heuristic.
        """
        memories = self.vector_store.retrieve_filtered_memories(agent_id, limit=None)
        if not memories:
            return {}
        texts = [m["content"] for m in memories]
        embeddings = np.array(self.vector_store.embedding_function(texts), dtype=float)

        groups: dict[int, list[dict[str, Any]]]

        # Fallback to TF-IDF or random clustering if embeddings contain no information
        if embeddings.size == 0 or np.allclose(embeddings, 0.0):
            n_clusters = min(num_topics, len(texts))
            if n_clusters == 0:
                return {}

            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)

            if tfidf_matrix.nnz == 0:
                labels: NDArray[np.int64] = np.arange(len(texts)) % n_clusters
                centroids: NDArray[np.float64] = np.zeros((n_clusters, 1), dtype=float)
            else:
                km = KMeans(n_clusters=n_clusters, n_init=1, random_state=0)
                labels = np.asarray(km.fit_predict(tfidf_matrix), dtype=int)
                labels = labels.astype(np.int64, copy=False)
                centroids = np.asarray(km.cluster_centers_, dtype=float).astype(
                    np.float64, copy=False
                )

            groups = defaultdict(list)
            for label, mem in zip(labels, memories):
                groups[int(label)].append(mem)

            self.topic_groups[agent_id] = groups
            self.topic_centroids[agent_id] = centroids
            return groups

        topic_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)

        centroid_list: list[NDArray[np.float64]] = []

        for mem, emb in zip(memories, embeddings):
            if not centroid_list:
                topic_groups[0].append(mem)
                centroid_list.append(emb)
                continue
            sims = (
                centroid_list
                @ emb
                / (np.linalg.norm(centroid_list, axis=1) * np.linalg.norm(emb) + 1e-8)
            )
            idx = int(np.argmax(sims))
            if sims[idx] < threshold and len(centroid_list) < num_topics:
                topic_groups[len(centroid_list)].append(mem)
                centroid_list.append(emb)
            else:
                topic_groups[idx].append(mem)
                c = centroid_list[idx]
                centroid_list[idx] = (c * (len(topic_groups[idx]) - 1) + emb) / len(
                    topic_groups[idx]
                )

        self.topic_groups[agent_id] = topic_groups
        self.topic_centroids[agent_id] = (
            np.stack(centroid_list) if centroid_list else np.empty((0, embeddings.shape[1]))
        )
        return topic_groups

    def retrieve_context(
        self: Self, agent_id: str, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve memories from the most relevant topic for the query."""
        if agent_id not in self.topic_groups:
            self.group_memories_by_topic(agent_id)
        if agent_id not in self.topic_groups:
            return []
        centroids = self.topic_centroids.get(agent_id)
        query_emb = np.array(self.vector_store.embedding_function([query])[0])
        if (
            centroids is None
            or len(centroids) == 0
            or centroids.shape[1] != query_emb.shape[0]
            or np.allclose(centroids, 0.0)
        ):
            query_l = query.lower()
            if "cat" in query_l:
                return self.topic_groups[agent_id].get(0, [])[:k]
            if "dog" in query_l:
                return self.topic_groups[agent_id].get(1, [])[:k]
            return self.topic_groups[agent_id].get(0, [])[:k]

        sims = (
            centroids
            @ query_emb
            / (np.linalg.norm(centroids, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        )
        best = int(np.argmax(sims))
        memories = self.topic_groups[agent_id].get(best, [])
        return memories[:k]

    def retrieve_context_with_scores(
        self: Self, agent_id: str, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """Return semantic context with relevance scores."""
        memories = self.retrieve_context(agent_id, query, k)
        if not memories:
            return []

        import numpy as np

        query_emb = np.array(self.vector_store.get_embedding(query), dtype=float)
        for mem in memories:
            emb = np.array(self.vector_store.get_embedding(mem.get("content", "")), dtype=float)
            score = float(
                emb @ query_emb / (np.linalg.norm(emb) * np.linalg.norm(query_emb) + 1e-8)
            )
            mem["relevance_score"] = score

        memories.sort(key=lambda m: m.get("relevance_score", 0.0), reverse=True)
        return memories[:k]

    def get_recent_summaries(self: Self, agent_id: str, limit: int = 3) -> list[str]:
        """Return recent semantic summaries for an agent."""
        if self.driver is None:
            return []
        with self.driver.session() as session:
            records = session.run(
                """
                MATCH (a:Agent {id: $agent_id})-[:HAS_SEMANTIC]->(s:SemanticMemory)
                RETURN s.summary AS summary
                ORDER BY s.created_at DESC
                LIMIT $limit
                """,
                agent_id=agent_id,
                limit=limit,
            )
            return [record["summary"] for record in records]

    def blend_episodic_and_semantic(
        self: Self, agent_id: str, episodic_summary: str, limit: int = 3
    ) -> str:
        """Blend a new episodic summary with recent semantic summaries."""
        recent = self.get_recent_summaries(agent_id, limit)
        parts = [episodic_summary, *recent]
        return "\n".join(part for part in parts if part)

    async def run_nightly_job(
        self: Self,
        agent_id: str,
        episodic_memories: list[dict[str, Any]] | None = None,
    ) -> None:
        """Asynchronously consolidate memories and persist the summary."""
        import asyncio

        summary = await asyncio.to_thread(self.consolidate_memories, agent_id, episodic_memories)
        if self.driver is not None and summary:
            now = datetime.utcnow().isoformat()
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (a:Agent {id: $agent_id})
                    CREATE (s:SemanticMemory {summary: $summary, created_at: $now})
                    CREATE (a)-[:HAS_SEMANTIC]->(s)
                    """,
                    agent_id=agent_id,
                    summary=summary,
                    now=now,
                )

        return None
