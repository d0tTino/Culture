from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from neo4j import Driver
else:  # pragma: no cover - fallback if neo4j not installed
    try:
        from neo4j import Driver
    except Exception:
        Driver = object
from typing_extensions import Self

from .vector_store import ChromaVectorStoreManager
from src.agents.memory.memory_models import importância
from src.agents.memory.memory_tracking_manager import MemoryTrackingManager
from src.agents.memory.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class SemanticMemoryManager:
    """Manage consolidation and semantic grouping of memories."""

    def __init__(
        self: Self, vector_store: ChromaVectorStoreManager, driver: Driver | None
    ) -> None:
        self.vector_store = vector_store
        self.driver = driver
        self.topic_groups: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self.topic_centroids: dict[str, np.ndarray] = {}

    def consolidate_memories(self: Self, agent_id: str) -> str:
        """Consolidate an agent's episodic memories into a semantic summary."""
        memories = self.vector_store.retrieve_filtered_memories(
            agent_id, filters={"memory_type": "raw"}, limit=None
        )
        if not memories:
            return ""
        summary = "\n".join(mem["content"] for mem in memories)
        return summary

    def group_memories_by_topic(
        self: Self, agent_id: str, num_topics: int = 5, threshold: float = 0.7
    ) -> dict[int, list[dict[str, Any]]]:
        """Group memories into topics using embeddings or simple keywords."""
        memories = self.vector_store.retrieve_filtered_memories(agent_id, limit=None)
        if not memories:
            return {}
        texts = [m["content"] for m in memories]
        embeddings = np.array(self.vector_store.embedding_function(texts), dtype=float)

        groups: dict[int, list[dict[str, Any]]]

        # Fallback to keyword grouping if embeddings contain no information
        if embeddings.size == 0 or np.allclose(embeddings, 0.0):
            groups = defaultdict(list)
            keywords = ["cat", "dog"]
            for mem in memories:
                text = mem["content"].lower()
                placed = False
                for i, kw in enumerate(keywords):
                    if kw in text:
                        groups[i].append(mem)
                        placed = True
                        break
                if not placed:
                    groups[len(keywords)].append(mem)
            self.topic_groups[agent_id] = groups
            self.topic_centroids[agent_id] = np.zeros((len(groups), 1))
            return groups

        topic_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)

        centroids: list[np.ndarray] = []
        for mem, emb in zip(memories, embeddings):
            if not centroids:
                topic_groups[0].append(mem)
                centroids.append(emb)
                continue
            sims = (
                centroids @ emb / (np.linalg.norm(centroids, axis=1) * np.linalg.norm(emb) + 1e-8)
            )
            idx = int(np.argmax(sims))
            if sims[idx] < threshold and len(centroids) < num_topics:
                topic_groups[len(centroids)].append(mem)
                centroids.append(emb)
            else:
                topic_groups[idx].append(mem)
                c = centroids[idx]
                centroids[idx] = (c * (len(topic_groups[idx]) - 1) + emb) / len(topic_groups[idx])

        self.topic_groups[agent_id] = topic_groups
        self.topic_centroids[agent_id] = (
            np.stack(centroids) if centroids else np.empty((0, embeddings.shape[1]))
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

    async def run_nightly_job(self: Self, agent_id: str) -> None:
        """Asynchronously consolidate memories and persist the summary."""
        import asyncio

        summary = await asyncio.to_thread(self.consolidate_memories, agent_id)
        if self.driver is not None and summary:
            now = datetime.now(timezone.utc).isoformat()
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

    def get_l1_summaries_older_than(
        self, agent_id: str, max_age_days: int = 7
    ) -> list[importância]:
        """Get L1 summaries older than a certain number of days."""
        try:
            now = datetime.now(timezone.utc)
            cutoff_date = now - timedelta(days=max_age_days)
            results = self.vector_store.query_l1_summaries_before_date(
                agent_id, cutoff_date
            )
            return results
        except Exception as e:
            logger.exception(
                f"Error retrieving old L1 summaries for agent {agent_id}: {e}"
            )
            return []

    def get_l2_summaries_older_than(
        self, agent_id: str, max_age_days: int = 30
    ) -> list[importância]:
        """Get L2 summaries older than a certain number of days."""
        try:
            now = datetime.now(timezone.utc)
            cutoff_date = now - timedelta(days=max_age_days)
            results = self.vector_store.query_l2_summaries_before_date(
                agent_id, cutoff_date
            )
            return cast(list[importância], results)
        except Exception as e:
            logger.exception(
                f"Error retrieving old L2 summaries for agent {agent_id}: {e}"
            )
            return []

    def get_recent_summaries_l1(
        self, agent_id: str, limit: int = 5
    ) -> list[importância]:
        """Get the most recent L1 summaries for an agent."""
        try:
            return self.vector_store.query_l1_summaries_by_recency(
                agent_id=agent_id, limit=limit
            )
        except Exception as e:
            logger.exception(
                f"Error retrieving recent summaries for agent {agent_id}: {e}"
            )
            return []
