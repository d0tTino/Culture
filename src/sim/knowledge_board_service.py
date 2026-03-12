from __future__ import annotations

import math
from collections import defaultdict
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
from src.sim.knowledge_board_queries import (
    AgentContributionQueryDTO,
    CausalChainQueryDTO,
    PagedQueryResultDTO,
    ProposalStatusQueryDTO,
    QueryFilters,
    RankedEntryDTO,
    RankingWeights,
    StoryDigestDTO,
    ThreadQueryDTO,
    TimelineQueryDTO,
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

    def query_timeline(self, query: TimelineQueryDTO) -> PagedQueryResultDTO:
        entries = self._filter_entries(self._all_entries(), query.filters)
        ranked = self._rank_entries(entries, query.ranking, anchor_entry_id=query.anchor_entry_id)
        return self._to_page(
            ranked, page=query.pagination.page, page_size=query.pagination.page_size
        )

    def query_thread(self, query: ThreadQueryDTO) -> PagedQueryResultDTO:
        all_entries = self._all_entries()
        children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in all_entries:
            parent_id = entry.get("parent_entry_id")
            if isinstance(parent_id, str) and parent_id:
                children_by_parent[parent_id].append(entry)
        queue = [query.root_entry_id]
        seen: set[str] = set()
        thread_entries: list[dict[str, Any]] = []
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            for child in children_by_parent.get(current, []):
                child_id = str(child.get("entry_id", ""))
                if child_id:
                    queue.append(child_id)
                thread_entries.append(child)
        ranked = self._rank_entries(
            thread_entries,
            query.ranking,
            anchor_entry_id=query.root_entry_id,
        )
        return self._to_page(
            ranked, page=query.pagination.page, page_size=query.pagination.page_size
        )

    def query_proposal_status(self, query: ProposalStatusQueryDTO) -> dict[str, Any]:
        entries = self._all_entries()
        proposal = next(
            (entry for entry in entries if str(entry.get("entry_id", "")) == query.proposal_id),
            None,
        )
        consensus = self.get_consensus_projection(query.proposal_id)
        votes = [
            entry
            for entry in entries
            if str(entry.get("parent_entry_id") or "") == query.proposal_id
            and str(entry.get("entry_type", "")).lower() == KnowledgeEntryType.VOTE.value
        ]
        ranked_votes = self._rank_entries(votes, RankingWeights())
        vote_page = self._to_page(
            ranked_votes,
            page=query.pagination.page,
            page_size=query.pagination.page_size,
        )
        return {
            "proposal_id": query.proposal_id,
            "proposal": self._entry_to_ranked(proposal, score=1.0, signals={}).to_dict()
            if proposal
            else None,
            "consensus": {
                "approvals": consensus.approvals,
                "rejections": consensus.rejections,
                "consensus": consensus.consensus,
            },
            "votes": vote_page.to_dict(),
        }

    def query_agent_contribution(self, query: AgentContributionQueryDTO) -> PagedQueryResultDTO:
        scoped_filters = QueryFilters(
            agent_id=query.agent_id,
            entry_types=query.filters.entry_types,
            tags=query.filters.tags,
            search=query.filters.search,
            start_step=query.filters.start_step,
            end_step=query.filters.end_step,
        )
        entries = self._filter_entries(self._all_entries(), scoped_filters)
        ranked = self._rank_entries(entries, query.ranking)
        return self._to_page(
            ranked, page=query.pagination.page, page_size=query.pagination.page_size
        )

    def query_causal_chain(self, query: CausalChainQueryDTO) -> list[RankedEntryDTO]:
        entries = self._all_entries()
        by_id = {str(entry.get("entry_id", "")): entry for entry in entries}
        path: list[RankedEntryDTO] = []
        current = by_id.get(query.entry_id)
        depth = max(1, query.depth)
        while current is not None and depth > 0:
            path.append(self._entry_to_ranked(current, score=1.0, signals={"causal_depth": depth}))
            parent_id = current.get("parent_entry_id")
            if not isinstance(parent_id, str) or not parent_id:
                break
            current = by_id.get(parent_id)
            depth -= 1
        return path

    def generate_story_digest(self, period: str) -> StoryDigestDTO:
        entries = self._all_entries()
        now = self._step_provider()
        window = 24 if period == "daily" else 168
        start_step = max(0, now - window)
        scoped = [entry for entry in entries if int(entry.get("step", 0)) >= start_step]
        ranked = self._rank_entries(scoped, RankingWeights())
        highlights = tuple(item.content_summary for item in ranked[:5])
        source_entry_ids = tuple(item.entry_id for item in ranked[:10])
        return StoryDigestDTO(
            period="daily" if period == "daily" else "weekly",
            start_step=start_step,
            end_step=now,
            generated_at_step=now,
            highlights=highlights,
            source_entry_ids=source_entry_ids,
        )

    def generate_story_digests(self) -> dict[str, StoryDigestDTO]:
        return {
            "daily": self.generate_story_digest("daily"),
            "weekly": self.generate_story_digest("weekly"),
        }

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

    def _all_entries(self) -> list[dict[str, Any]]:
        get_full_entries = getattr(self._board, "get_full_entries", None)
        if callable(get_full_entries):
            return list(get_full_entries())
        return []

    def _filter_entries(
        self, entries: list[dict[str, Any]], filters: QueryFilters
    ) -> list[dict[str, Any]]:
        filtered = list(entries)
        if filters.agent_id:
            filtered = [
                entry for entry in filtered if str(entry.get("agent_id", "")) == filters.agent_id
            ]
        if filters.entry_types:
            accepted_types = {entry_type.lower() for entry_type in filters.entry_types}
            filtered = [
                entry
                for entry in filtered
                if str(entry.get("entry_type", "")).lower() in accepted_types
            ]
        if filters.tags:
            accepted_tags = {tag.lower() for tag in filters.tags}
            filtered = [
                entry
                for entry in filtered
                if accepted_tags.intersection(
                    {str(tag).lower() for tag in (entry.get("tags") or []) if isinstance(tag, str)}
                )
            ]
        if filters.search:
            needle = filters.search.strip().lower()
            filtered = [
                entry
                for entry in filtered
                if needle in str(entry.get("content_full", "")).lower()
                or needle in str(entry.get("content_summary", "")).lower()
            ]
        if filters.start_step is not None:
            filtered = [
                entry for entry in filtered if int(entry.get("step", 0)) >= filters.start_step
            ]
        if filters.end_step is not None:
            filtered = [
                entry for entry in filtered if int(entry.get("step", 0)) <= filters.end_step
            ]
        return filtered

    def _rank_entries(
        self,
        entries: list[dict[str, Any]],
        weights: RankingWeights,
        *,
        anchor_entry_id: str | None = None,
    ) -> list[RankedEntryDTO]:
        if not entries:
            return []
        now = self._step_provider()
        max_gap = max(1, max(now - int(entry.get("step", 0)) for entry in entries))
        vote_agg = self._aggregate_votes(entries)
        total_support = max(1, sum(row.get("support", 0) for row in vote_agg.values()))
        anchor = next(
            (
                entry
                for entry in entries
                if str(entry.get("entry_id", "")) == (anchor_entry_id or "")
            ),
            None,
        )
        anchor_agent = str(anchor.get("agent_id", "")) if anchor is not None else ""

        scored: list[RankedEntryDTO] = []
        for entry in entries:
            step = int(entry.get("step", 0))
            recency = 1.0 - min(1.0, max(0, now - step) / max_gap)
            support = self._support_for_entry(entry, vote_agg)
            endorsement = min(1.0, support / total_support)
            proximity = self._relationship_proximity(entry, anchor_agent)
            score = (
                weights.recency * recency
                + weights.endorsement * endorsement
                + weights.proximity * proximity
            )
            scored.append(
                self._entry_to_ranked(
                    entry,
                    score=score,
                    signals={
                        "recency": recency,
                        "endorsement": endorsement,
                        "proximity": proximity,
                    },
                )
            )
        scored.sort(key=lambda item: (item.score, item.step), reverse=True)
        return scored

    def _aggregate_votes(self, entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        aggregate_votes = getattr(self._board, "aggregate_votes", None)
        if callable(aggregate_votes):
            return aggregate_votes(None)
        result: dict[str, dict[str, int]] = {}
        for entry in entries:
            if str(entry.get("entry_type", "")).lower() != KnowledgeEntryType.VOTE.value:
                continue
            proposal_id = str(entry.get("parent_entry_id") or "")
            if not proposal_id:
                continue
            row = result.setdefault(proposal_id, {"approvals": 0, "rejections": 0, "support": 0})
            approve = bool((entry.get("reference_metadata") or {}).get("approve", False))
            if approve:
                row["approvals"] += 1
                row["support"] += 1
            else:
                row["rejections"] += 1
        return result

    def _support_for_entry(
        self,
        entry: dict[str, Any],
        vote_agg: dict[str, dict[str, int]],
    ) -> int:
        entry_id = str(entry.get("entry_id", ""))
        if not entry_id:
            return 0
        if str(entry.get("entry_type", "")).lower() == KnowledgeEntryType.VOTE.value:
            metadata = entry.get("reference_metadata") or {}
            return 1 if bool(metadata.get("approve", False)) else 0
        return int(vote_agg.get(entry_id, {}).get("support", 0))

    def _relationship_proximity(self, entry: dict[str, Any], anchor_agent: str) -> float:
        if not anchor_agent:
            return 0.5
        if str(entry.get("agent_id", "")) == anchor_agent:
            return 1.0
        parent_id = entry.get("parent_entry_id")
        return 0.75 if isinstance(parent_id, str) and parent_id else 0.4

    def _entry_to_ranked(
        self,
        entry: dict[str, Any] | None,
        *,
        score: float,
        signals: dict[str, float],
    ) -> RankedEntryDTO:
        if entry is None:
            return RankedEntryDTO(
                entry_id="",
                step=0,
                agent_id="",
                entry_type="",
                content_summary="",
                parent_entry_id=None,
                tags=(),
                score=score,
                signals=signals,
            )
        summary = str(entry.get("content_summary") or entry.get("content_full") or "")
        clipped = summary[:240]
        return RankedEntryDTO(
            entry_id=str(entry.get("entry_id", "")),
            step=int(entry.get("step", 0)),
            agent_id=str(entry.get("agent_id", "")),
            entry_type=str(entry.get("entry_type", "")),
            content_summary=clipped,
            parent_entry_id=entry.get("parent_entry_id"),
            tags=tuple(str(tag) for tag in (entry.get("tags") or []) if isinstance(tag, str)),
            score=math.floor(score * 1000) / 1000,
            signals={k: math.floor(v * 1000) / 1000 for k, v in signals.items()},
        )

    def _to_page(
        self, entries: list[RankedEntryDTO], *, page: int, page_size: int
    ) -> PagedQueryResultDTO:
        normalized_page = max(1, page)
        normalized_page_size = max(1, page_size)
        offset = (normalized_page - 1) * normalized_page_size
        return PagedQueryResultDTO(
            total=len(entries),
            page=normalized_page,
            page_size=normalized_page_size,
            items=tuple(entries[offset : offset + normalized_page_size]),
        )

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
