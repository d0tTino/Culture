"""Graph-backed implementation of the Knowledge Board."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from neo4j import Driver, GraphDatabase
from typing_extensions import Self

from src.infra import config
from src.interfaces import metrics

logger = logging.getLogger(__name__)


class GraphKnowledgeBoard:
    """Knowledge Board backed by a Neo4j graph database."""

    def __init__(
        self: Self,
        *,
        driver: Driver | None = None,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        if driver is not None:
            self.driver = driver
        else:
            self.driver = GraphDatabase.driver(
                uri or config.GRAPH_DB_URI,
                auth=(user or config.GRAPH_DB_USER, password or config.GRAPH_DB_PASSWORD),
            )
        metrics.KNOWLEDGE_BOARD_SIZE.set(self._count_entries())

    # --- Internal helpers -------------------------------------------------
    def _run(self: Self, query: str, **params: Any) -> list[Any]:
        with self.driver.session() as session:
            result = session.run(query, **params)
            return list(result)

    def _count_entries(self: Self) -> int:
        res = self._run("MATCH (e:KBEntry) RETURN count(e) AS cnt")
        return int(res[0]["cnt"]) if res else 0

    # --- Public API -------------------------------------------------------
    def get_state(self: Self, max_entries: int = 10) -> list[str]:
        records = self._run(
            "MATCH (e:KBEntry) RETURN e ORDER BY e.step DESC LIMIT $limit",
            limit=max_entries,
        )
        entries = [rec["e"] for rec in records]
        # Return in chronological order like the in-memory board
        entries = list(reversed(entries))
        return [entry["content_display"] for entry in entries]

    def get_full_entries(self: Self) -> list[dict[str, Any]]:
        records = self._run("MATCH (e:KBEntry) RETURN e ORDER BY e.step ASC")
        return [dict(record["e"]) for record in records]

    def to_dict(self: Self) -> dict[str, Any]:
        return {"entries": self.get_full_entries()}

    def get_recent_entries_for_prompt(self: Self, max_entries: int = 5) -> list[str]:
        records = self._run(
            "MATCH (e:KBEntry) RETURN e ORDER BY e.step DESC LIMIT $limit",
            limit=max_entries,
        )
        entries = [rec["e"] for rec in records]
        entries = list(reversed(entries))
        if not entries:
            return ["(Knowledge Board is empty)"]
        formatted_entries = []
        for entry in entries:
            step = entry.get("step", "N/A")
            agent_id = entry.get("agent_id", "Unknown Agent")
            content_summary = entry.get("content_summary", entry.get("content_full", "N/A"))
            max_content_len = 150
            if len(content_summary) > max_content_len:
                content_summary = content_summary[:max_content_len] + "..."
            formatted_entries.append(f"[Step {step}, {agent_id}]: {content_summary}")
        return formatted_entries

    def add_entry(self: Self, entry: str, agent_id: str, step: int) -> bool:
        entry_id = str(uuid.uuid4())
        formatted_content = f"Step {step} (Agent: {agent_id}): {entry}"
        props = {
            "entry_id": entry_id,
            "step": step,
            "agent_id": agent_id,
            "content_full": entry,
            "content_display": formatted_content,
            "content_summary": entry,
        }
        self._run("CREATE (e:KBEntry $props)", props=props)
        metrics.KNOWLEDGE_BOARD_SIZE.set(self._count_entries())
        logger.info(
            "GraphKnowledgeBoard: Added entry %s by %s at step %s", entry_id, agent_id, step
        )
        return True

    def clear_board(self: Self) -> None:
        self._run("MATCH (e:KBEntry) DETACH DELETE e")
        metrics.KNOWLEDGE_BOARD_SIZE.set(0)
