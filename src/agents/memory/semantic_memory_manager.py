from __future__ import annotations

import logging
from datetime import datetime

from neo4j import Driver
from typing_extensions import Self

from .vector_store import ChromaVectorStoreManager

logger = logging.getLogger(__name__)


class SemanticMemoryManager:
    """Manage consolidation of episodic memories into semantic memories."""

    def __init__(self: Self, vector_store: ChromaVectorStoreManager, driver: Driver) -> None:
        self.vector_store = vector_store
        self.driver = driver

    def consolidate_memories(self: Self, agent_id: str) -> str:
        """Consolidate an agent's episodic memories into a semantic summary."""
        memories = self.vector_store.retrieve_filtered_memories(
            agent_id, filters={"memory_type": "raw"}, limit=None
        )
        if not memories:
            return ""
        summary = "\n".join(mem["content"] for mem in memories)
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
        return summary

    def get_recent_summaries(self: Self, agent_id: str, limit: int = 3) -> list[str]:
        """Return recent semantic summaries for an agent."""
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
        """Asynchronously consolidate memories, intended to run nightly."""
        import asyncio

        await asyncio.to_thread(self.consolidate_memories, agent_id)
