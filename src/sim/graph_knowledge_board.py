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


from typing_extensions import Self

from src.infra import config
from src.interfaces import metrics
from src.sim.knowledge_board import BoardEntry, prepare_entry_payload
from src.sim.knowledge_entry import (
    KnowledgeEntryType,
    KnowledgeRelationshipType,
    migrate_entry_dict,
)

logger = logging.getLogger(__name__)


class GraphKnowledgeBoard:
    supports_threads = True
    supports_causal_chain = True
    supports_votes = True
    supports_graph_queries = True

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

    def __enter__(self: Self) -> Self:  # pragma: no cover
        return self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:  # pragma: no cover
        self.close()

    def _run(self: Self, query: str, **params: Any) -> list[Any]:
        with self.driver.session() as session:
            result = session.run(query, **params)
            return list(result)

    def _count_entries(self: Self) -> int:
        res = self._run("MATCH (e:KBEntry) RETURN count(e) AS cnt")
        return int(res[0]["cnt"]) if res else 0

    def begin_transaction(self: Self) -> dict[str, Any]:
        """Capture a snapshot used for compensating rollback."""

        return self.to_snapshot()

    def commit_transaction(self: Self, tx_context: object) -> None:
        """Finalize a transaction context (graph backend commits per-query)."""

        _ = tx_context

    def rollback_transaction(self: Self, tx_context: object) -> None:
        """Restore graph state from a captured snapshot."""

        if isinstance(tx_context, dict):
            self.from_snapshot(tx_context)

    def get_state(self: Self, max_entries: int = 10) -> list[str]:
        records = self._run(
            "MATCH (e:KBEntry) RETURN e ORDER BY e.step DESC LIMIT $limit",
            limit=max_entries,
        )
        entries = [rec["e"] for rec in records]
        entries = list(reversed(entries))
        return [entry["content_display"] for entry in entries]

    def get_full_entries(self: Self) -> list[dict[str, Any]]:
        records = self._run("MATCH (e:KBEntry) RETURN e ORDER BY e.step ASC")
        return [dict(record["e"]) for record in records]

    def to_dict(self: Self) -> dict[str, Any]:
        return self.to_snapshot()

    def to_snapshot(self: Self) -> dict[str, Any]:
        return {"entries": self.get_full_entries()}

    def from_snapshot(self: Self, snapshot: dict[str, Any]) -> None:
        entries = snapshot.get("entries", [])
        if isinstance(entries, list):
            self.replace_entries(
                [migrate_entry_dict(entry) for entry in entries if isinstance(entry, dict)]
            )
        else:
            self.replace_entries([])

    def replace_entries(self: Self, entries: list[dict[str, Any]]) -> None:
        self.clear_board()
        for entry in entries:
            props = migrate_entry_dict(entry)
            agent_id = str(props.get("agent_id", "unknown"))
            self._run(
                """
                MERGE (a:Agent {agent_id: $agent_id})
                CREATE (e:KBEntry)
                SET e = $props
                MERGE (a)-[:AUTHORED]->(e)
                """,
                agent_id=agent_id,
                props=props,
            )
            self._create_typed_relationships(agent_id=agent_id, props=props)
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
        entry: BoardEntry,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        entry_id, record = prepare_entry_payload(entry, agent_id, step)
        props = record.to_dict()
        self._run(
            """
            MERGE (a:Agent {agent_id: $agent_id})
            CREATE (e:KBEntry)
            SET e = $props
            MERGE (a)-[:AUTHORED]->(e)
            """,
            agent_id=agent_id,
            props=props,
        )
        self._create_typed_relationships(agent_id=agent_id, props=props)
        metrics.KNOWLEDGE_BOARD_SIZE.set(self._count_entries())
        logger.info(
            "GraphKnowledgeBoard: Added entry %s by %s at step %s", entry_id, agent_id, step
        )
        return True

    def _create_typed_relationships(self: Self, *, agent_id: str, props: dict[str, Any]) -> None:
        entry_id = str(props.get("entry_id", ""))
        parent_entry_id = props.get("parent_entry_id")
        if parent_entry_id:
            relation = KnowledgeRelationshipType.AMENDS.value
            metadata = props.get("reference_metadata") or {}
            relationship_hint = str(metadata.get("relationship") or "").lower()
            if relationship_hint == "supersedes":
                relation = KnowledgeRelationshipType.SUPERCEDES.value
            self._run(
                f"""
                MATCH (source:KBEntry {{entry_id: $entry_id}})
                MATCH (target:KBEntry {{entry_id: $target_id}})
                MERGE (source)-[:{relation}]->(target)
                """,
                entry_id=entry_id,
                target_id=str(parent_entry_id),
            )

        if props.get("entry_type") == KnowledgeEntryType.VOTE.value and parent_entry_id:
            metadata = props.get("reference_metadata") or {}
            approve = bool(metadata.get("approve", False))
            self._run(
                """
                MERGE (a:Agent {agent_id: $agent_id})
                MATCH (p:KBEntry {entry_id: $proposal_id})
                MERGE (a)-[v:VOTED]->(p)
                SET v.approve = $approve, v.vote_entry_id = $vote_entry_id
                """,
                agent_id=agent_id,
                proposal_id=str(parent_entry_id),
                approve=approve,
                vote_entry_id=entry_id,
            )

        if props.get("entry_type") == KnowledgeEntryType.ENDORSEMENT.value and parent_entry_id:
            self._run(
                """
                MERGE (a:Agent {agent_id: $agent_id})
                MATCH (target:KBEntry {entry_id: $target_id})
                MERGE (a)-[:ENDORSED]->(target)
                """,
                agent_id=agent_id,
                target_id=str(parent_entry_id),
            )

    def record_vote(self: Self, *, voter_agent_id: str, proposal_id: str, approve: bool) -> None:
        self._run(
            """
            MERGE (a:Agent {agent_id: $voter_agent_id})
            MATCH (p:KBEntry {entry_id: $proposal_id})
            MERGE (a)-[v:VOTED]->(p)
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
            OPTIONAL MATCH (:Agent)-[e:ENDORSED]->(i)
            WITH i, count(e) AS endorsements
            WHERE endorsements >= $min_endorsements
            RETURN i AS entry, endorsements
            ORDER BY endorsements DESC, i.step DESC
            LIMIT $limit
            """,
            min_endorsements=min_endorsements,
            limit=limit,
        )
        return [
            {**dict(record["entry"]), "endorsement_count": int(record["endorsements"])}
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

    def get_active_proposals(self: Self, limit: int = 20) -> list[dict[str, Any]]:
        records = self._run(
            """
            MATCH (p:KBEntry {entry_type: 'proposal'})
            WHERE NOT EXISTS { MATCH (:KBEntry)-[:SUPERCEDES]->(p) }
            RETURN p AS proposal
            ORDER BY p.step DESC
            LIMIT $limit
            """,
            limit=limit,
        )
        return [dict(record["proposal"]) for record in records]

    def get_consensus_status(self: Self, proposal_id: str) -> dict[str, Any]:
        records = self._run(
            """
            MATCH (p:KBEntry {entry_id: $proposal_id})
            OPTIONAL MATCH (:Agent)-[v:VOTED]->(p)
            WITH p, sum(CASE WHEN v.approve THEN 1 ELSE 0 END) AS approvals,
                 sum(CASE WHEN v.approve THEN 0 ELSE 1 END) AS rejections
            RETURN p.entry_id AS proposal_id, approvals, rejections
            """,
            proposal_id=proposal_id,
        )
        if not records:
            return {
                "proposal_id": proposal_id,
                "approvals": 0,
                "rejections": 0,
                "consensus": False,
            }
        row = records[0]
        approvals = int(row["approvals"] or 0)
        rejections = int(row["rejections"] or 0)
        return {
            "proposal_id": row["proposal_id"],
            "approvals": approvals,
            "rejections": rejections,
            "consensus": approvals > rejections,
        }

    def get_agent_stance_history(self: Self, agent_id: str) -> list[dict[str, Any]]:
        records = self._run(
            """
            MATCH (a:Agent {agent_id: $agent_id})-[:AUTHORED]->(e:KBEntry)
            WHERE e.entry_type IN ['vote', 'endorsement']
            RETURN e.entry_id AS entry_id,
                   e.step AS step,
                   e.entry_type AS entry_type,
                   e.parent_entry_id AS parent_entry_id,
                   e.target_agent_id AS target_agent_id
            ORDER BY e.step ASC
            """,
            agent_id=agent_id,
        )
        return [dict(record) for record in records]

    def get_active_proposal_projection(self: Self, limit: int = 20) -> list[Any]:
        from src.sim.knowledge_board_protocol import ActiveProposalProjection

        return [
            ActiveProposalProjection(
                entry_id=str(item.get("entry_id", "")),
                step=int(item.get("step", 0)),
                agent_id=str(item.get("agent_id", "")),
                content_summary=str(item.get("content_summary") or item.get("content_full") or ""),
            )
            for item in self.get_active_proposals(limit)
        ]

    def get_consensus_projection(self: Self, proposal_id: str) -> Any:
        from src.sim.knowledge_board_protocol import ConsensusStatusProjection

        row = self.get_consensus_status(proposal_id)
        return ConsensusStatusProjection(
            proposal_id=str(row.get("proposal_id", proposal_id)),
            approvals=int(row.get("approvals", 0)),
            rejections=int(row.get("rejections", 0)),
            consensus=bool(row.get("consensus", False)),
        )

    def get_agent_stance_projection(self: Self, agent_id: str) -> list[Any]:
        from src.sim.knowledge_board_protocol import AgentStanceProjection

        return [
            AgentStanceProjection(
                entry_id=str(item.get("entry_id", "")),
                step=int(item.get("step", 0)),
                entry_type=str(item.get("entry_type", "")),
                parent_entry_id=item.get("parent_entry_id"),
                target_agent_id=item.get("target_agent_id"),
                stance=item.get("stance"),
            )
            for item in self.get_agent_stance_history(agent_id)
        ]

    def _get_endorsement_count(self: Self, entry_id: str) -> int:
        if not entry_id:
            return 0
        records = self._run(
            """
            MATCH (e:KBEntry {entry_id: $entry_id})
            OPTIONAL MATCH (:Agent)-[v:ENDORSED]->(e)
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
                entry_type=KnowledgeEntryType.PROPOSAL,
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
        close_fn = getattr(self.driver, "close", None)
        if callable(close_fn):
            close_fn()
