"""Graph-backed implementation of the Knowledge Board."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

try:  # pragma: no cover - optional dependency
    from neo4j import Driver, GraphDatabase
except Exception:  # pragma: no cover - handle missing package
    Driver = object  # type: ignore[misc]

    class GraphDatabase:  # type: ignore[no-redef]
        @staticmethod
        def driver(*_a: object, **_k: object) -> None:
            raise RuntimeError("neo4j not installed")


try:  # pragma: no cover - optional dependency
    from neo4j.exceptions import Neo4jError
except Exception:  # pragma: no cover - handle missing package
    Neo4jError = Exception
from typing_extensions import Self

from src.infra import config
from src.interfaces import metrics
from src.sim.knowledge_board import BoardEntry, prepare_entry_payload

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
        self.lock = asyncio.Lock()
        metrics.KNOWLEDGE_BOARD_SIZE.set(self._count_entries())

    # Enable use as a context manager
    def __enter__(self: Self) -> Self:  # pragma: no cover - convenience
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:  # pragma: no cover - convenience
        self.close()

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
        return self.to_snapshot()

    def to_snapshot(self: Self) -> dict[str, Any]:
        """Serialize backend state required to restore this board."""

        return {"entries": self.get_full_entries()}

    def from_snapshot(self: Self, snapshot: dict[str, Any]) -> None:
        """Restore board state from a serialized snapshot."""

        entries = snapshot.get("entries", [])
        if isinstance(entries, list):
            self.replace_entries([entry for entry in entries if isinstance(entry, dict)])
        else:
            self.replace_entries([])

    def replace_entries(self: Self, entries: list[dict[str, Any]]) -> None:
        """Replace graph-backed KB entries from a serialized snapshot."""
        self.clear_board()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            props = dict(entry)
            agent_id = str(props.get("agent_id", "unknown"))
            entry_type = str(props.get("entry_type", "note"))
            self._run(
                """
                MERGE (a:Agent {agent_id: $agent_id})
                CREATE (e:KBEntry)
                SET e = $props
                SET e.entry_type = $entry_type
                MERGE (a)-[:AUTHORED]->(e)
                """,
                agent_id=agent_id,
                props=props,
                entry_type=entry_type,
            )
            self._create_reference_links(str(props.get("entry_id", "")), props.get("reference_metadata"))
        metrics.KNOWLEDGE_BOARD_SIZE.set(self._count_entries())

    def get_recent_entries_for_prompt(
        self: Self,
        max_entries: int = 5,
        *,
        include_relationship_summaries: bool = False,
    ) -> list[str]:
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
            relationship_summary = ""
            if include_relationship_summaries:
                endorsement_count = self._get_endorsement_count(entry.get("entry_id", ""))
                relationship_summary = f" (endorsements: {endorsement_count})"
            formatted_entries.append(
                f"[Step {step}, {agent_id}]: {content_summary}{relationship_summary}"
            )
        return formatted_entries

    def add_entry(
        self: Self,
        entry: str | BoardEntry,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        entry_id, props = prepare_entry_payload(entry, agent_id, step)
        self._run(
            """
            MERGE (a:Agent {agent_id: $agent_id})
            CREATE (e:KBEntry)
            SET e = $props
            SET e.entry_type = $entry_type
            MERGE (a)-[:AUTHORED]->(e)
            """,
            agent_id=agent_id,
            props=props,
            entry_type=props["entry_type"],
        )
        self._create_reference_links(entry_id, props.get("reference_metadata"))
        metrics.KNOWLEDGE_BOARD_SIZE.set(self._count_entries())
        logger.info(
            "GraphKnowledgeBoard: Added entry %s by %s at step %s",
            props["entry_id"],
            agent_id,
            step,
        )
        return True

    def record_vote(
        self: Self,
        *,
        voter_agent_id: str,
        proposal_id: str,
        approve: bool,
    ) -> None:
        self._run(
            """
            MERGE (a:Agent {agent_id: $voter_agent_id})
            MATCH (p:KBEntry {entry_id: $proposal_id})
            MERGE (a)-[v:VOTED {proposal_id: $proposal_id}]->(p)
            SET v.approve = $approve
            """,
            voter_agent_id=voter_agent_id,
            proposal_id=proposal_id,
            approve=approve,
        )

    def get_proposal_support_counts(
        self: Self, proposal_ids: list[str] | None = None
    ) -> dict[str, int]:
        records = self._run(
            """
            MATCH (p:KBEntry)
            WHERE p.entry_type = 'proposal'
            AND ($proposal_ids IS NULL OR p.entry_id IN $proposal_ids)
            OPTIONAL MATCH (:Agent)-[v:VOTED {approve: true}]->(p)
            RETURN p.entry_id AS proposal_id, count(v) AS support_count
            """,
            proposal_ids=proposal_ids,
        )
        return {record["proposal_id"]: int(record["support_count"]) for record in records}

    def get_endorsed_ideas(
        self: Self,
        *,
        min_endorsements: int = 1,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        records = self._run(
            """
            MATCH (i:KBEntry {entry_type: 'idea'})
            OPTIONAL MATCH (:Agent)-[v:VOTED {approve: true}]->(i)
            WITH i, count(v) AS endorsements
            WHERE endorsements >= $min_endorsements
            RETURN i AS entry, endorsements
            ORDER BY endorsements DESC, i.step DESC
            LIMIT $limit
            """,
            min_endorsements=min_endorsements,
            limit=limit,
        )
        return [
            {
                **dict(record["entry"]),
                "endorsement_count": int(record["endorsements"]),
            }
            for record in records
        ]

    def get_agent_contribution_graph(
        self: Self, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        records = self._run(
            """
            MATCH (a:Agent)-[:AUTHORED]->(e:KBEntry)
            WHERE $agent_id IS NULL OR a.agent_id = $agent_id
            RETURN a.agent_id AS agent_id, e.entry_id AS entry_id, e.entry_type AS entry_type, e.step AS step
            ORDER BY e.step ASC
            """,
            agent_id=agent_id,
        )
        return [dict(record) for record in records]

    def _create_reference_links(self: Self, entry_id: str, reference_metadata: Any) -> None:
        if not isinstance(reference_metadata, dict):
            return
        references = reference_metadata.get("references")
        if not isinstance(references, list):
            return
        normalized_references = [str(reference_id) for reference_id in references if reference_id]
        if not normalized_references:
            return
        self._run(
            """
            MATCH (source:KBEntry {entry_id: $entry_id})
            UNWIND $references AS target_id
            MATCH (target:KBEntry {entry_id: target_id})
            MERGE (source)-[:REFERENCES]->(target)
            """,
            entry_id=entry_id,
            references=normalized_references,
        )

    def _get_endorsement_count(self: Self, entry_id: str) -> int:
        if not entry_id:
            return 0
        records = self._run(
            """
            MATCH (e:KBEntry {entry_id: $entry_id})
            OPTIONAL MATCH (:Agent)-[v:VOTED {approve: true}]->(e)
            RETURN count(v) AS endorsements
            """,
            entry_id=entry_id,
        )
        return int(records[0]["endorsements"]) if records else 0

    def add_law_proposal(
        self: Self,
        proposal: str,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        return self.add_entry(
            BoardEntry(
                content_full=f"Law proposed: {proposal}",
                entry_type="proposal",
                tags=["governance", "proposal"],
            ),
            agent_id,
            step,
            vector,
        )

    def clear_board(self: Self) -> None:
        self._run("MATCH (n) WHERE n:KBEntry OR n:Agent DETACH DELETE n")
        metrics.KNOWLEDGE_BOARD_SIZE.set(0)

    def close(self: Self) -> None:
        """Close the underlying Neo4j driver."""
        try:
            self.driver.close()
        except Neo4jError as exc:  # pragma: no cover - defensive
            logger.exception("GraphKnowledgeBoard: failed to close driver: %s", exc)
