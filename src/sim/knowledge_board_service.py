from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.governance.knowledge_governance_transaction import (
    GovernanceMutation,
    KnowledgeGovernanceTransaction,
    make_proposal_idempotency_key,
    make_vote_idempotency_key,
)
from src.sim.knowledge_board_protocol import (
    ActiveProposalProjection,
    AgentStanceProjection,
    ConsensusStatusProjection,
    KnowledgeBoardProtocol,
)
from src.sim.knowledge_entry import KnowledgeEntry, KnowledgeEntryType


class KnowledgeBoardService:
    """Backend-agnostic mediator for knowledge board reads/writes."""

    def __init__(
        self,
        board: KnowledgeBoardProtocol,
        *,
        step_provider: Callable[[], int],
        vector_provider: Callable[[], dict[str, int]],
    ) -> None:
        self._board = board
        self._step_provider = step_provider
        self._vector_provider = vector_provider
        self._transaction = KnowledgeGovernanceTransaction(board, step_provider=step_provider)

    def get_recent_entries_for_prompt(self, max_entries: int = 5) -> list[str]:
        return self._board.get_recent_entries_for_prompt(max_entries=max_entries)

    def get_active_proposal_projection(self, limit: int = 20) -> list[ActiveProposalProjection]:
        return self._board.get_active_proposal_projection(limit=limit)

    def get_consensus_projection(self, proposal_id: str) -> ConsensusStatusProjection:
        return self._board.get_consensus_projection(proposal_id)

    def get_agent_stance_projection(self, agent_id: str) -> list[AgentStanceProjection]:
        return self._board.get_agent_stance_projection(agent_id)

    async def post_idea(
        self,
        *,
        actor_id: str,
        content: str,
        causal_source: str,
        tags: list[str] | None = None,
        reference_metadata: dict[str, Any] | None = None,
        governance_rule_id: str | None = None,
    ) -> bool:
        return await self._post_entry(
            actor_id=actor_id,
            entry=KnowledgeEntry(
                content_full=content,
                entry_type=KnowledgeEntryType.IDEA,
                tags=self._merge_tags(["idea", "proposal"], tags),
                reference_metadata=self._with_required_metadata(
                    reference_metadata,
                    actor_id=actor_id,
                    causal_source=causal_source,
                    governance_rule_id=governance_rule_id,
                ),
                governance_rule_id=governance_rule_id,
            ),
        )

    async def post_proposal(
        self,
        *,
        actor_id: str,
        content: str,
        causal_source: str,
        tags: list[str] | None = None,
        reference_metadata: dict[str, Any] | None = None,
        governance_rule_id: str | None = None,
    ) -> bool:
        step = self._step_provider()
        idempotency_key = make_proposal_idempotency_key(
            proposer_id=actor_id,
            text=content,
            step=step,
        )
        metadata = dict(reference_metadata or {})
        metadata["idempotency_key"] = idempotency_key
        entry = KnowledgeEntry(
            content_full=content,
            entry_type=KnowledgeEntryType.PROPOSAL,
            tags=self._merge_tags(["governance", "proposal"], tags),
            reference_metadata=self._with_required_metadata(
                metadata,
                actor_id=actor_id,
                causal_source=causal_source,
                governance_rule_id=governance_rule_id,
            ),
            governance_rule_id=governance_rule_id,
        )
        return await self._transaction.execute(
            actor_id=actor_id,
            board_entry=entry,
            idempotency_key=idempotency_key,
            governance_mutation=GovernanceMutation(apply=lambda: None, rollback=lambda _t: None),
        )

    async def post_vote(
        self,
        *,
        actor_id: str,
        proposal_id: str,
        approve: bool,
        causal_source: str,
        tags: list[str] | None = None,
        reference_metadata: dict[str, Any] | None = None,
        governance_rule_id: str | None = None,
    ) -> bool:
        step = self._step_provider()
        idempotency_key = make_vote_idempotency_key(
            voter_id=actor_id,
            proposal_entry_id=proposal_id,
            approve=approve,
            step=step,
        )
        metadata = dict(reference_metadata or {})
        metadata["approve"] = bool(approve)
        metadata["proposal_id"] = proposal_id
        metadata["idempotency_key"] = idempotency_key
        entry = KnowledgeEntry(
            content_full=(
                f"Vote by {actor_id} on proposal {proposal_id}: {'approve' if approve else 'reject'}"
            ),
            entry_type=KnowledgeEntryType.VOTE,
            parent_entry_id=proposal_id,
            tags=self._merge_tags(["governance", "vote"], tags),
            reference_metadata=self._with_required_metadata(
                metadata,
                actor_id=actor_id,
                causal_source=causal_source,
                governance_rule_id=governance_rule_id,
            ),
            governance_rule_id=governance_rule_id,
        )
        return await self._transaction.execute(
            actor_id=actor_id,
            board_entry=entry,
            idempotency_key=idempotency_key,
            governance_mutation=GovernanceMutation(apply=lambda: None, rollback=lambda _t: None),
        )

    async def post_event(
        self,
        *,
        actor_id: str,
        content: str,
        causal_source: str,
        event_type: KnowledgeEntryType | str = KnowledgeEntryType.EVENT,
        tags: list[str] | None = None,
        reference_metadata: dict[str, Any] | None = None,
        governance_rule_id: str | None = None,
    ) -> bool:
        return await self._post_entry(
            actor_id=actor_id,
            entry=KnowledgeEntry(
                content_full=content,
                entry_type=event_type,
                tags=self._merge_tags(["event"], tags),
                reference_metadata=self._with_required_metadata(
                    reference_metadata,
                    actor_id=actor_id,
                    causal_source=causal_source,
                    governance_rule_id=governance_rule_id,
                ),
                governance_rule_id=governance_rule_id,
            ),
        )

    async def post_lifecycle_transition(
        self,
        *,
        actor_id: str,
        from_state: str,
        to_state: str,
        reason: str,
        legacy_artifacts: list[str] | None,
        causal_source: str,
        tags: list[str] | None = None,
        reference_metadata: dict[str, Any] | None = None,
    ) -> bool:
        metadata = dict(reference_metadata or {})
        metadata.update(
            {
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
                "legacy_artifacts": legacy_artifacts or [],
            }
        )
        return await self._post_entry(
            actor_id=actor_id,
            entry=KnowledgeEntry(
                content_full=f"Agent {actor_id} transitioned lifecycle {from_state} -> {to_state}",
                entry_type=KnowledgeEntryType.RETIREMENT,
                tags=self._merge_tags(["population", "retirement", to_state], tags),
                reference_metadata=self._with_required_metadata(
                    metadata,
                    actor_id=actor_id,
                    causal_source=causal_source,
                ),
            ),
        )

    async def post_human_message(
        self,
        *,
        actor_id: str,
        content: str,
        causal_source: str,
        tags: list[str] | None = None,
        reference_metadata: dict[str, Any] | None = None,
    ) -> bool:
        return await self._post_entry(
            actor_id=actor_id,
            entry=KnowledgeEntry(
                content_full=content,
                entry_type=KnowledgeEntryType.HUMAN_MESSAGE,
                tags=self._merge_tags(["human"], tags),
                reference_metadata=self._with_required_metadata(
                    reference_metadata,
                    actor_id=actor_id,
                    causal_source=causal_source,
                ),
            ),
        )

    async def _post_entry(self, *, actor_id: str, entry: KnowledgeEntry) -> bool:
        step = self._step_provider()
        lock = getattr(self._board, "lock", None)
        if lock is not None:
            async with lock:
                return self._board.add_entry(entry, actor_id, step, self._vector_provider())
        return self._board.add_entry(entry, actor_id, step, self._vector_provider())

    def _with_required_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        actor_id: str,
        causal_source: str,
        governance_rule_id: str | None = None,
    ) -> dict[str, Any]:
        base = dict(metadata or {})
        provenance_raw = base.get("provenance")
        provenance: dict[str, Any] = (
            dict(provenance_raw) if isinstance(provenance_raw, dict) else {}
        )
        base["provenance"] = {
            **provenance,
            "step": self._step_provider(),
            "actor": actor_id,
            "causal_source": causal_source,
        }
        if governance_rule_id:
            governance_raw = base.get("governance")
            governance: dict[str, Any] = (
                dict(governance_raw) if isinstance(governance_raw, dict) else {}
            )
            base["governance"] = {
                **governance,
                "rule_id": governance_rule_id,
            }
        return base

    @staticmethod
    def _merge_tags(default_tags: list[str], tags: list[str] | None) -> list[str]:
        merged: list[str] = []
        for tag in [*default_tags, *(tags or [])]:
            if tag and tag not in merged:
                merged.append(tag)
        return merged
