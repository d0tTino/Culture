"""Protocol definitions and adapters for knowledge board backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from src.sim.graph_knowledge_board import GraphKnowledgeBoard
from src.sim.knowledge_board import KnowledgeBoard
from src.sim.knowledge_entry import KnowledgeEntry


@dataclass(frozen=True)
class TimeRange:
    start_step: int | None = None
    end_step: int | None = None


@dataclass(frozen=True)
class KnowledgeQuery:
    semantic: str | None = None
    topics: tuple[str, ...] = ()
    time_range: TimeRange | None = None
    limit: int = 20


@dataclass(frozen=True)
class ActiveProposalProjection:
    entry_id: str
    step: int
    agent_id: str
    content_summary: str


@dataclass(frozen=True)
class ConsensusStatusProjection:
    proposal_id: str
    approvals: int
    rejections: int
    consensus: bool


@dataclass(frozen=True)
class AgentStanceProjection:
    entry_id: str
    step: int
    entry_type: str
    parent_entry_id: str | None
    target_agent_id: str | None
    stance: str | None


@runtime_checkable
class KnowledgeBoardCapabilities(Protocol):
    """Strict capability contract used by callers."""

    lock: Any
    supports_threads: bool
    supports_causal_chain: bool
    supports_votes: bool
    supports_graph_queries: bool

    def append_entry(
        self,
        entry: KnowledgeEntry,
        *,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool: ...

    def link_entries(
        self,
        *,
        source_entry_id: str,
        target_entry_id: str,
        relationship: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool: ...

    def query_entries(self, query: KnowledgeQuery) -> list[dict[str, Any]]: ...

    def aggregate_votes(
        self, proposal_ids: list[str] | None = None
    ) -> dict[str, dict[str, int]]: ...

    def begin_transaction(self) -> object: ...

    def commit_transaction(self, tx_context: object) -> None: ...

    def rollback_transaction(self, tx_context: object) -> None: ...

    def to_snapshot(self) -> dict[str, Any]: ...

    def from_snapshot(self, snapshot: dict[str, Any]) -> None: ...

    def replace_entries(self, entries: list[dict[str, Any]]) -> None: ...

    def get_recent_entries_for_prompt(self, max_entries: int = 5) -> list[str]: ...

    def get_active_proposals(self, limit: int = 20) -> list[dict[str, Any]]: ...

    def get_consensus_status(self, proposal_id: str) -> dict[str, Any]: ...

    def get_agent_stance_history(self, agent_id: str) -> list[dict[str, Any]]: ...

    def get_active_proposal_projection(
        self, limit: int = 20
    ) -> list[ActiveProposalProjection]: ...

    def get_consensus_projection(self, proposal_id: str) -> ConsensusStatusProjection: ...

    def get_agent_stance_projection(self, agent_id: str) -> list[AgentStanceProjection]: ...

    def record_vote(self, *, voter_agent_id: str, proposal_id: str, approve: bool) -> None: ...


# Backward-compatible alias.
KnowledgeBoardProtocol = KnowledgeBoardCapabilities
EntryStore = KnowledgeBoardCapabilities
SemanticQueryStore = KnowledgeBoardCapabilities
ProposalVotingStore = KnowledgeBoardCapabilities
RelationshipStore = KnowledgeBoardCapabilities
TransactionalKnowledgeBoardProtocol = KnowledgeBoardCapabilities


class KnowledgeBoardCapabilityError(RuntimeError):
    """Base error for capability resolution failures."""


class UnsupportedKnowledgeBoardCapabilityError(KnowledgeBoardCapabilityError):
    """Raised when a required capability is unavailable for a board backend."""

    def __init__(self, *, capability: str, board: object) -> None:
        board_type = type(board).__name__
        super().__init__(
            f"Knowledge board '{board_type}' does not support required capability '{capability}'."
        )
        self.capability = capability
        self.board_type = board_type


CapabilityT = TypeVar("CapabilityT")


def _require_capability(board: object, capability: type[CapabilityT], name: str) -> CapabilityT:
    if isinstance(board, capability):
        return cast(CapabilityT, board)
    raise UnsupportedKnowledgeBoardCapabilityError(capability=name, board=board)


def _coerce_board(board: object) -> KnowledgeBoardCapabilities:
    if isinstance(board, KnowledgeBoardCapabilities):
        return cast(KnowledgeBoardCapabilities, board)
    if isinstance(board, GraphKnowledgeBoard):
        return GraphKnowledgeBoardAdapter(board)
    if isinstance(board, KnowledgeBoard):
        return InMemoryKnowledgeBoardAdapter(board)
    raise UnsupportedKnowledgeBoardCapabilityError(
        capability="KnowledgeBoardCapabilities", board=board
    )


def as_entry_store(board: object) -> EntryStore:
    return _coerce_board(board)


def as_relationship_store(board: object) -> RelationshipStore:
    capability_board = _coerce_board(board)
    if not getattr(capability_board, "supports_graph_queries", False):
        raise UnsupportedKnowledgeBoardCapabilityError(capability="RelationshipStore", board=board)
    return capability_board


def as_proposal_voting_store(board: object) -> ProposalVotingStore:
    capability_board = _coerce_board(board)
    if not getattr(capability_board, "supports_votes", False):
        raise UnsupportedKnowledgeBoardCapabilityError(
            capability="ProposalVotingStore", board=board
        )
    return capability_board


def as_semantic_query_store(board: object) -> SemanticQueryStore:
    return _coerce_board(board)


def _normalize_entries(
    entries: list[dict[str, Any]], query: KnowledgeQuery
) -> list[dict[str, Any]]:
    filtered = entries
    if query.topics:
        topics = {topic.lower() for topic in query.topics}
        filtered = [
            entry
            for entry in filtered
            if topics.intersection({str(tag).lower() for tag in entry.get("tags", [])})
        ]
    if query.time_range is not None:
        start = query.time_range.start_step
        end = query.time_range.end_step
        filtered = [
            entry
            for entry in filtered
            if (start is None or int(entry.get("step", -1)) >= start)
            and (end is None or int(entry.get("step", -1)) <= end)
        ]
    if query.semantic:
        needle = query.semantic.lower().strip()
        filtered = [
            entry
            for entry in filtered
            if needle in str(entry.get("content_full", "")).lower()
            or needle in str(entry.get("content_summary", "")).lower()
        ]
    return filtered[-max(1, query.limit) :]


def _aggregate_vote_entries(
    entries: list[dict[str, Any]], proposal_ids: list[str] | None = None
) -> dict[str, dict[str, int]]:
    allowed = set(proposal_ids or []) if proposal_ids else None
    result: dict[str, dict[str, int]] = {}
    for entry in entries:
        if entry.get("entry_type") != "vote":
            continue
        proposal_id = str(entry.get("parent_entry_id") or "")
        if not proposal_id or (allowed is not None and proposal_id not in allowed):
            continue
        row = result.setdefault(proposal_id, {"approvals": 0, "rejections": 0, "support": 0})
        approve = bool((entry.get("reference_metadata") or {}).get("approve", False))
        if approve:
            row["approvals"] += 1
            row["support"] += 1
        else:
            row["rejections"] += 1
    return result


class InMemoryKnowledgeBoardAdapter:
    """Adapter that exposes strict capability contract for in-memory board."""

    def __init__(self, board: KnowledgeBoard | None = None) -> None:
        self._board = board or KnowledgeBoard()
        self.lock = self._board.lock
        self.supports_threads = self._board.supports_threads
        self.supports_causal_chain = self._board.supports_causal_chain
        self.supports_votes = self._board.supports_votes
        self.supports_graph_queries = self._board.supports_graph_queries
        self._links: list[dict[str, Any]] = []

    def add_entry(
        self,
        entry: KnowledgeEntry,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        return self.append_entry(entry, agent_id=agent_id, step=step, vector=vector)

    def append_entry(
        self,
        entry: KnowledgeEntry,
        *,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        return self._board.add_entry(entry, agent_id=agent_id, step=step, vector=vector)

    def link_entries(
        self,
        *,
        source_entry_id: str,
        target_entry_id: str,
        relationship: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        self._links.append(
            {
                "source": source_entry_id,
                "target": target_entry_id,
                "relationship": relationship.lower(),
                "metadata": dict(metadata or {}),
            }
        )
        return True

    def query_entries(self, query: KnowledgeQuery) -> list[dict[str, Any]]:
        return _normalize_entries(self._board.get_full_entries(), query)

    def aggregate_votes(self, proposal_ids: list[str] | None = None) -> dict[str, dict[str, int]]:
        return _aggregate_vote_entries(self._board.get_full_entries(), proposal_ids)

    def record_vote(self, *, voter_agent_id: str, proposal_id: str, approve: bool) -> None:
        self.append_entry(
            KnowledgeEntry(
                content_full=f"Vote by {voter_agent_id} on {proposal_id}",
                entry_type="vote",
                parent_entry_id=proposal_id,
                reference_metadata={"approve": approve},
            ),
            agent_id=voter_agent_id,
            step=0,
        )

    def get_proposal_support_counts(self, proposal_ids: list[str] | None = None) -> dict[str, int]:
        return {pid: vals["support"] for pid, vals in self.aggregate_votes(proposal_ids).items()}

    def begin_transaction(self) -> object:
        return {"snapshot": self.to_snapshot()}

    def commit_transaction(self, tx_context: object) -> None:
        _ = tx_context

    def rollback_transaction(self, tx_context: object) -> None:
        if isinstance(tx_context, dict) and isinstance(tx_context.get("snapshot"), dict):
            self.from_snapshot(cast(dict[str, Any], tx_context["snapshot"]))

    def get_recent_entries_for_prompt(self, max_entries: int = 5) -> list[str]:
        return self._board.get_recent_entries_for_prompt(max_entries=max_entries)

    def get_full_entries(self) -> list[dict[str, Any]]:
        return self._board.get_full_entries()

    def replace_entries(self, entries: list[dict[str, Any]]) -> None:
        self._board.replace_entries(entries)

    def to_snapshot(self) -> dict[str, Any]:
        snapshot = self._board.to_snapshot()
        snapshot["links"] = list(self._links)
        return snapshot

    def from_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._board.from_snapshot(snapshot)
        links = snapshot.get("links", [])
        self._links = [dict(link) for link in links if isinstance(link, dict)]

    def get_active_proposals(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query_entries(KnowledgeQuery(topics=("proposal",), limit=limit))

    def get_consensus_status(self, proposal_id: str) -> dict[str, Any]:
        agg = self.aggregate_votes([proposal_id]).get(
            proposal_id, {"approvals": 0, "rejections": 0}
        )
        return {
            "proposal_id": proposal_id,
            "approvals": agg["approvals"],
            "rejections": agg["rejections"],
            "consensus": agg["approvals"] > agg["rejections"],
        }

    def get_agent_stance_history(self, agent_id: str) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self._board.get_full_entries()
            if entry.get("agent_id") == agent_id
            and entry.get("entry_type") in {"vote", "endorsement"}
        ]

    def get_active_proposal_projection(self, limit: int = 20) -> list[ActiveProposalProjection]:
        return [
            ActiveProposalProjection(
                entry_id=str(item.get("entry_id", "")),
                step=int(item.get("step", 0)),
                agent_id=str(item.get("agent_id", "")),
                content_summary=str(item.get("content_summary") or item.get("content_full") or ""),
            )
            for item in self._board.get_active_proposals(limit)
        ]

    def get_consensus_projection(self, proposal_id: str) -> ConsensusStatusProjection:
        row = self.get_consensus_status(proposal_id)
        return ConsensusStatusProjection(
            proposal_id=str(row.get("proposal_id", proposal_id)),
            approvals=int(row.get("approvals", 0)),
            rejections=int(row.get("rejections", 0)),
            consensus=bool(row.get("consensus", False)),
        )

    def get_agent_stance_projection(self, agent_id: str) -> list[AgentStanceProjection]:
        return [
            AgentStanceProjection(
                entry_id=str(item.get("entry_id", "")),
                step=int(item.get("step", 0)),
                entry_type=str(item.get("entry_type", "")),
                parent_entry_id=item.get("parent_entry_id"),
                target_agent_id=item.get("target_agent_id"),
                stance=(
                    (item.get("reference_metadata") or {}).get("stance")
                    if isinstance(item.get("reference_metadata"), dict)
                    else item.get("stance")
                ),
            )
            for item in self.get_agent_stance_history(agent_id)
        ]


class GraphKnowledgeBoardAdapter(InMemoryKnowledgeBoardAdapter):
    """Adapter for graph board with operation-journal rollback semantics."""

    def __init__(self, board: GraphKnowledgeBoard | None = None) -> None:
        self._graph = board or GraphKnowledgeBoard()
        self.lock = self._graph.lock
        self.supports_threads = self._graph.supports_threads
        self.supports_causal_chain = self._graph.supports_causal_chain
        self.supports_votes = self._graph.supports_votes
        self.supports_graph_queries = self._graph.supports_graph_queries
        self._tx_journal: dict[int, dict[str, list[dict[str, Any]]]] = {}

    def append_entry(
        self,
        entry: KnowledgeEntry,
        *,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        ok = self._graph.add_entry(entry, agent_id=agent_id, step=step, vector=vector)
        if ok and self._tx_journal:
            added = self._graph.get_full_entries()[-1]
            for journal in self._tx_journal.values():
                journal.setdefault("entries", []).append(
                    {"entry_id": str(added.get("entry_id", ""))}
                )
        return ok

    def add_entry(
        self,
        entry: KnowledgeEntry,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        return self.append_entry(entry, agent_id=agent_id, step=step, vector=vector)

    def link_entries(
        self,
        *,
        source_entry_id: str,
        target_entry_id: str,
        relationship: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        rel = relationship.upper()
        self._graph._run(
            f"""
            MATCH (source:KBEntry {{entry_id: $source_entry_id}})
            MATCH (target:KBEntry {{entry_id: $target_entry_id}})
            MERGE (source)-[r:{rel}]->(target)
            SET r += $metadata
            """,
            source_entry_id=source_entry_id,
            target_entry_id=target_entry_id,
            metadata=dict(metadata or {}),
        )
        for journal in self._tx_journal.values():
            journal.setdefault("links", []).append(
                {"source": source_entry_id, "target": target_entry_id, "relationship": rel}
            )
        return True

    def query_entries(self, query: KnowledgeQuery) -> list[dict[str, Any]]:
        return _normalize_entries(self._graph.get_full_entries(), query)

    def aggregate_votes(self, proposal_ids: list[str] | None = None) -> dict[str, dict[str, int]]:
        return _aggregate_vote_entries(self._graph.get_full_entries(), proposal_ids)

    def record_vote(self, *, voter_agent_id: str, proposal_id: str, approve: bool) -> None:
        self._graph.record_vote(
            voter_agent_id=voter_agent_id, proposal_id=proposal_id, approve=approve
        )

    def get_proposal_support_counts(self, proposal_ids: list[str] | None = None) -> dict[str, int]:
        return self._graph.get_proposal_support_counts(proposal_ids)

    def get_endorsed_ideas(
        self,
        *,
        min_endorsements: int = 1,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self._graph.get_endorsed_ideas(
            min_endorsements=min_endorsements,
            limit=limit,
        )

    def get_agent_contribution_graph(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        return self._graph.get_agent_contribution_graph(agent_id=agent_id)

    def begin_transaction(self) -> object:
        tx_id = id(object())
        self._tx_journal[tx_id] = {"entries": [], "links": []}
        return tx_id

    def commit_transaction(self, tx_context: object) -> None:
        if isinstance(tx_context, int):
            self._tx_journal.pop(tx_context, None)

    def rollback_transaction(self, tx_context: object) -> None:
        if not isinstance(tx_context, int):
            return
        journal = self._tx_journal.pop(tx_context, None)
        if journal is None:
            return
        for link in reversed(journal.get("links", [])):
            self._graph._run(
                f"""
                MATCH (source:KBEntry {{entry_id: $source}})-[r:{link["relationship"]}]->(target:KBEntry {{entry_id: $target}})
                DELETE r
                """,
                source=link["source"],
                target=link["target"],
            )
        for entry in reversed(journal.get("entries", [])):
            self._graph._run(
                "MATCH (e:KBEntry {entry_id: $entry_id}) DETACH DELETE e",
                entry_id=entry.get("entry_id", ""),
            )

    def get_recent_entries_for_prompt(self, max_entries: int = 5) -> list[str]:
        return self._graph.get_recent_entries_for_prompt(max_entries=max_entries)

    def replace_entries(self, entries: list[dict[str, Any]]) -> None:
        self._graph.replace_entries(entries)

    def to_snapshot(self) -> dict[str, Any]:
        return self._graph.to_snapshot()

    def from_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._graph.from_snapshot(snapshot)

    def get_active_proposals(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._graph.get_active_proposals(limit=limit)

    def get_consensus_status(self, proposal_id: str) -> dict[str, Any]:
        return self._graph.get_consensus_status(proposal_id)

    def get_agent_stance_history(self, agent_id: str) -> list[dict[str, Any]]:
        return self._graph.get_agent_stance_history(agent_id)

    def get_active_proposal_projection(self, limit: int = 20) -> list[ActiveProposalProjection]:
        return [
            ActiveProposalProjection(
                entry_id=str(item.get("entry_id", "")),
                step=int(item.get("step", 0)),
                agent_id=str(item.get("agent_id", "")),
                content_summary=str(item.get("content_summary") or item.get("content_full") or ""),
            )
            for item in self._graph.get_active_proposals(limit)
        ]

    def get_consensus_projection(self, proposal_id: str) -> ConsensusStatusProjection:
        row = self._graph.get_consensus_status(proposal_id)
        return ConsensusStatusProjection(
            proposal_id=str(row.get("proposal_id", proposal_id)),
            approvals=int(row.get("approvals", 0)),
            rejections=int(row.get("rejections", 0)),
            consensus=bool(row.get("consensus", False)),
        )

    def get_agent_stance_projection(self, agent_id: str) -> list[AgentStanceProjection]:
        return [
            AgentStanceProjection(
                entry_id=str(item.get("entry_id", "")),
                step=int(item.get("step", 0)),
                entry_type=str(item.get("entry_type", "")),
                parent_entry_id=item.get("parent_entry_id"),
                target_agent_id=item.get("target_agent_id"),
                stance=item.get("stance"),
            )
            for item in self._graph.get_agent_stance_history(agent_id)
        ]


class VectorAugmentedKnowledgeBoardAdapter(InMemoryKnowledgeBoardAdapter):
    """Optional in-memory adapter with simple vector similarity hook."""

    def __init__(self, board: KnowledgeBoard | None = None) -> None:
        super().__init__(board=board)
        self._semantic_vectors: dict[str, set[str]] = {}

    def append_entry(
        self,
        entry: KnowledgeEntry,
        *,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool:
        ok = super().append_entry(entry, agent_id=agent_id, step=step, vector=vector)
        if ok:
            latest = self.get_full_entries()[-1]
            text = str(latest.get("content_full", "")).lower()
            self._semantic_vectors[str(latest.get("entry_id", ""))] = set(text.split())
        return ok


def supports_voting(board: object) -> bool:
    return bool(getattr(board, "supports_votes", False))


def supports_graph_queries(board: object) -> bool:
    return bool(getattr(board, "supports_graph_queries", False))


def supports_read_models(board: object) -> bool:
    return hasattr(board, "get_active_proposals") and hasattr(board, "get_consensus_status")
