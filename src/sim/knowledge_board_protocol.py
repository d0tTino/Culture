"""Protocol definitions and capability adapters for knowledge board backends."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from src.sim.knowledge_entry import KnowledgeEntry


@runtime_checkable
class KnowledgeBoardProtocol(Protocol):
    """Core interface required by simulation knowledge board backends."""

    def add_entry(
        self,
        entry: str | KnowledgeEntry,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool: ...

    def replace_entries(self, entries: list[dict[str, Any]]) -> None: ...

    def to_snapshot(self) -> dict[str, Any]: ...

    def from_snapshot(self, snapshot: dict[str, Any]) -> None: ...

    def get_recent_entries_for_prompt(self, max_entries: int = 5) -> list[str]: ...


@runtime_checkable
class EntryStore(KnowledgeBoardProtocol, Protocol):
    """Capability for core knowledge entry storage and retrieval."""


@runtime_checkable
class RelationshipStore(Protocol):
    """Capability for relationship-level graph projections."""

    def get_endorsed_ideas(
        self,
        *,
        min_endorsements: int = 1,
        limit: int = 20,
    ) -> list[dict[str, Any]]: ...

    def get_agent_contribution_graph(
        self, agent_id: str | None = None
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class ProposalVotingStore(Protocol):
    """Capability for proposal voting persistence and support queries."""

    def record_vote(self, *, voter_agent_id: str, proposal_id: str, approve: bool) -> None: ...

    def get_proposal_support_counts(
        self, proposal_ids: list[str] | None = None
    ) -> dict[str, int]: ...


@runtime_checkable
class SemanticQueryStore(Protocol):
    """Capability for governance-oriented read models used by agents/UI."""

    def get_active_proposals(self, limit: int = 20) -> list[dict[str, Any]]: ...

    def get_consensus_status(self, proposal_id: str) -> dict[str, Any]: ...

    def get_agent_stance_history(self, agent_id: str) -> list[dict[str, Any]]: ...


@runtime_checkable
class TransactionalKnowledgeBoardProtocol(Protocol):
    """Capability for transactional writes with explicit rollback hooks."""

    def begin_transaction(self) -> object: ...

    def commit_transaction(self, tx_context: object) -> None: ...

    def rollback_transaction(self, tx_context: object) -> None: ...


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


def as_entry_store(board: object) -> EntryStore:
    """Adapt ``board`` to :class:`EntryStore` or fail explicitly."""

    return _require_capability(board, EntryStore, "EntryStore")


def as_relationship_store(board: object) -> RelationshipStore:
    """Adapt ``board`` to :class:`RelationshipStore` or fail explicitly."""

    return _require_capability(board, RelationshipStore, "RelationshipStore")


def as_proposal_voting_store(board: object) -> ProposalVotingStore:
    """Adapt ``board`` to :class:`ProposalVotingStore` or fail explicitly."""

    return _require_capability(board, ProposalVotingStore, "ProposalVotingStore")


def as_semantic_query_store(board: object) -> SemanticQueryStore:
    """Adapt ``board`` to :class:`SemanticQueryStore` or fail explicitly."""

    return _require_capability(board, SemanticQueryStore, "SemanticQueryStore")


@runtime_checkable
class KnowledgeBoardVoteProtocol(Protocol):
    """Optional voting extension supported by graph-backed boards."""

    def record_vote(self, *, voter_agent_id: str, proposal_id: str, approve: bool) -> None: ...

    def get_proposal_support_counts(
        self, proposal_ids: list[str] | None = None
    ) -> dict[str, int]: ...


@runtime_checkable
class KnowledgeBoardGraphProtocol(Protocol):
    """Optional graph query extension supported by graph-backed boards."""

    def get_endorsed_ideas(
        self,
        *,
        min_endorsements: int = 1,
        limit: int = 20,
    ) -> list[dict[str, Any]]: ...

    def get_agent_contribution_graph(
        self, agent_id: str | None = None
    ) -> list[dict[str, Any]]: ...


def supports_voting(board: object) -> bool:
    """Return ``True`` when the board supports voting extension APIs."""

    return isinstance(board, ProposalVotingStore)


def supports_graph_queries(board: object) -> bool:
    """Return ``True`` when the board supports graph query extension APIs."""

    return isinstance(board, RelationshipStore)


@runtime_checkable
class KnowledgeBoardReadModelProtocol(Protocol):
    """Optional read-model extension used by UI and agent views."""

    def get_active_proposals(self, limit: int = 20) -> list[dict[str, Any]]: ...

    def get_consensus_status(self, proposal_id: str) -> dict[str, Any]: ...

    def get_agent_stance_history(self, agent_id: str) -> list[dict[str, Any]]: ...


def supports_read_models(board: object) -> bool:
    """Return ``True`` when board supports governance read-model APIs."""

    return isinstance(board, SemanticQueryStore)
