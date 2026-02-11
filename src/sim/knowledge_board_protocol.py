"""Protocol definitions for knowledge board backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.sim.knowledge_board import BoardEntry


@runtime_checkable
class KnowledgeBoardProtocol(Protocol):
    """Core interface required by simulation knowledge board backends."""

    def add_entry(
        self,
        entry: str | BoardEntry,
        agent_id: str,
        step: int,
        vector: dict[str, int] | None = None,
    ) -> bool: ...

    def replace_entries(self, entries: list[dict[str, Any]]) -> None: ...

    def to_snapshot(self) -> dict[str, Any]: ...

    def from_snapshot(self, snapshot: dict[str, Any]) -> None: ...

    def get_recent_entries_for_prompt(self, max_entries: int = 5) -> list[str]: ...


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

    return isinstance(board, KnowledgeBoardVoteProtocol)


def supports_graph_queries(board: object) -> bool:
    """Return ``True`` when the board supports graph query extension APIs."""

    return isinstance(board, KnowledgeBoardGraphProtocol)

