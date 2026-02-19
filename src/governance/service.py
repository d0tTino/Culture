from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast

from typing_extensions import Self

from src.agents.core.base_agent import Agent
from src.infra.ledger import ledger
from src.sim.knowledge_board_protocol import KnowledgeBoardProtocol, supports_voting
from src.sim.knowledge_entry import KnowledgeEntry, KnowledgeEntryType
from src.utils.policy import evaluate_with_opa

from .law_board import law_board
from .rules_engine import governance_rules_engine


def quadratic_vote_weight(ip_balance: float, staked_ip: float = 0.0) -> float:
    """Return the quadratic voting weight from available influence points.

    The weight is computed as ``sqrt(ip_balance + staked_ip)``. Any negative
    totals are treated as ``0`` to avoid ``ValueError`` from ``sqrt``.
    """
    return math.sqrt(max(0.0, ip_balance + staked_ip))


class GovernanceService:
    """Service coordinating law proposals and voting."""

    def __init__(self) -> None:
        self._knowledge_board: KnowledgeBoardProtocol | None = None
        self._step_provider: Callable[[], int] = lambda: 0

    def attach_knowledge_board(
        self,
        board: KnowledgeBoardProtocol | None,
        *,
        step_provider: Callable[[], int] | None = None,
    ) -> None:
        """Attach a board so governance writes become queryable provenance entries."""
        self._knowledge_board = board
        if step_provider is not None:
            self._step_provider = step_provider

    def _current_step(self) -> int:
        try:
            return int(self._step_provider())
        except Exception:
            return 0

    def _write_governance_entry(
        self,
        *,
        agent_id: str,
        entry: KnowledgeEntry,
    ) -> None:
        board = self._knowledge_board
        if board is None:
            return
        try:
            board.add_entry(entry, agent_id, self._current_step())
        except Exception:
            return

    async def vote(self: Self, agent: Agent, proposal: str) -> bool:
        """Return the agent's vote (True=approve) using the OPA policy."""
        allowed, _ = await evaluate_with_opa(proposal)
        return allowed

    async def vote_weighted(
        self: Self,
        agent: Agent,
        proposal: str,
        weight: int = 1,
        approve: bool = True,
        proposal_entry_id: str | None = None,
        governance_rule_id: str | None = None,
    ) -> bool:
        """Cast a weighted ``approve`` or ``reject`` vote for ``proposal``."""
        if weight < 1:
            weight = 1
        allowed = await self.vote(agent, proposal)
        if not allowed:
            return False
        cost = float(weight**2)
        try:
            await ledger.spend(agent.agent_id, ip=cost, reason="vote")
        except Exception:
            return False

        vote_entry = KnowledgeEntry(
            content_full=f"{'Approve' if approve else 'Reject'} vote for: {proposal}",
            entry_type=KnowledgeEntryType.VOTE,
            tags=["governance", "vote"],
            parent_entry_id=proposal_entry_id,
            governance_rule_id=governance_rule_id,
            reference_metadata={
                "approve": approve,
                "weight": weight,
                "stance": "approve" if approve else "reject",
            },
        )
        self._write_governance_entry(agent_id=agent.agent_id, entry=vote_entry)
        if proposal_entry_id and self._knowledge_board and supports_voting(self._knowledge_board):
            self._knowledge_board.record_vote(
                voter_agent_id=agent.agent_id,
                proposal_id=proposal_entry_id,
                approve=approve,
            )
        return approve

    async def stake_ip(self: Self, agent_id: str, amount: float) -> float:
        """Stake ``amount`` of IP for ``agent_id`` and return total staked IP."""

        def _stake() -> float:
            ledger.stake_ip(agent_id, amount)
            try:
                return ledger.get_staked_ip(agent_id)
            except Exception:
                return 0.0

        return await asyncio.to_thread(_stake)

    async def submit_proposal(
        self: Self,
        proposer: Agent,
        text: str,
        agents: Iterable[Agent],
        vote_weights: dict[str, int] | None = None,
    ) -> dict[str, float | bool | dict[str, Any] | str] | bool:
        """Convenience wrapper around :meth:`propose_law`."""
        return await self.propose_law(proposer, text, agents, vote_weights)

    async def propose_law(
        self: Self,
        proposer: Agent,
        text: str,
        agents: Iterable[Agent],
        vote_weights: dict[str, int] | None = None,
    ) -> dict[str, float | bool | dict[str, Any] | str] | bool:
        """Propose ``text`` to ``agents`` and persist the vote outcome."""
        allowed, _ = await evaluate_with_opa(text)
        if not allowed:
            return False

        proposal_payload = KnowledgeEntry(
            content_full=text,
            entry_type=KnowledgeEntryType.PROPOSAL,
            tags=["governance", "proposal"],
        )
        proposal_entry_id = ""
        if self._knowledge_board is not None:
            from src.sim.knowledge_board import prepare_entry_payload

            proposal_entry_id, _ = prepare_entry_payload(
                proposal_payload,
                proposer.agent_id,
                self._current_step(),
            )
            self._write_governance_entry(agent_id=proposer.agent_id, entry=proposal_payload)

        votes = await asyncio.gather(*[self.vote(a, text) for a in agents])
        weights: list[float] = []
        ip_spent = 0.0
        if vote_weights is None:
            for a in agents:
                base_ip = getattr(a.state, "ip", 0.0)
                try:
                    staked = ledger.get_staked_ip(a.agent_id)
                except Exception:
                    staked = 0.0
                weights.append(quadratic_vote_weight(base_ip, staked))
        else:
            balance_tasks: list[Awaitable[tuple[float, float]]] = [
                ledger.get_balance_async(a.agent_id) for a in agents
            ]
            start_balances = await asyncio.gather(*balance_tasks, return_exceptions=True)
            spend_tasks: list[Awaitable[tuple[float, float]]] = []
            for a in agents:
                w = int(vote_weights.get(a.agent_id, 1))
                weights.append(float(w))
                cost = float(w**2)
                spend_tasks.append(ledger.spend(a.agent_id, ip=cost, reason="vote"))
            end_balances = await asyncio.gather(*spend_tasks, return_exceptions=True)
            for before, after in zip(start_balances, end_balances):
                if isinstance(before, Exception) or isinstance(after, Exception):
                    continue
                before_bal = cast(tuple[float, float], before)
                after_bal = cast(tuple[float, float], after)

                ip_spent += max(0.0, before_bal[0] - after_bal[0])

        yes_weight = sum(w for w, v in zip(weights, votes) if v)
        no_weight = sum(w for w, v in zip(weights, votes) if not v)
        approved = yes_weight > no_weight
        if approved:
            law_board.add_law(text)
        outcome: dict[str, float | bool | dict[str, Any] | str] = {
            "approved": approved,
            "yes_weight": yes_weight,
            "no_weight": no_weight,
            "ip_spent": ip_spent,
            "proposal_entry_id": proposal_entry_id,
        }
        rule_materialization = governance_rules_engine.materialize_from_proposal(
            text,
            proposer_id=proposer.agent_id,
            approved=approved,
            proposal_record=outcome,
        )
        outcome["rule_materialization"] = rule_materialization
        governance_rule_id = (
            str(rule_materialization.get("rule_id"))
            if isinstance(rule_materialization, dict)
            else None
        )

        if self._knowledge_board is not None:
            for agent, vote in zip(agents, votes):
                self._write_governance_entry(
                    agent_id=agent.agent_id,
                    entry=KnowledgeEntry(
                        content_full=f"{'Approve' if vote else 'Reject'} vote for proposal: {text}",
                        entry_type=KnowledgeEntryType.VOTE,
                        parent_entry_id=proposal_entry_id or None,
                        governance_rule_id=governance_rule_id,
                        tags=["governance", "vote"],
                        reference_metadata={
                            "approve": vote,
                            "stance": "approve" if vote else "reject",
                        },
                    ),
                )
                if proposal_entry_id and supports_voting(self._knowledge_board):
                    self._knowledge_board.record_vote(
                        voter_agent_id=agent.agent_id,
                        proposal_id=proposal_entry_id,
                        approve=vote,
                    )

            if approved:
                self._write_governance_entry(
                    agent_id=proposer.agent_id,
                    entry=KnowledgeEntry(
                        content_full=f"Law ratified: {text}",
                        entry_type=KnowledgeEntryType.LAW,
                        parent_entry_id=proposal_entry_id or None,
                        governance_rule_id=governance_rule_id,
                        tags=["governance", "law"],
                        reference_metadata={"relationship": "supersedes"},
                    ),
                )

        try:
            ledger.record_law_proposal(
                proposer.agent_id,
                text,
                outcome["approved"],
                outcome["yes_weight"],
                outcome["no_weight"],
                outcome["ip_spent"],
            )
        except Exception:
            pass
        return outcome

    def get_proposals(self: Self, limit: int | None = None) -> list[dict[str, object]]:
        """Return stored law proposals."""
        try:
            return ledger.get_law_proposals(limit)
        except Exception:
            return []


governance = GovernanceService()

__all__ = ["GovernanceService", "governance", "quadratic_vote_weight"]
