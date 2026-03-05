from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast

from typing_extensions import Self

from src.agents.core.base_agent import Agent
from src.governance.knowledge_governance_transaction import (
    GovernanceMutation,
    KnowledgeGovernanceTransaction,
    make_proposal_idempotency_key,
    make_vote_idempotency_key,
)
from src.infra.ledger import ledger
from src.sim.knowledge_board_protocol import EntryStore
from src.sim.knowledge_entry import KnowledgeEntry, KnowledgeEntryType
from src.utils.policy import evaluate_with_opa

from .law_board import law_board
from .rules_engine import governance_rules_engine


async def _emit_governance_event(payload: dict[str, Any]) -> None:
    from src.interfaces.dashboard_backend import SimulationEvent, emit_event

    event_type = str(payload.get("type", "governance"))
    await emit_event(SimulationEvent(type=event_type, data=payload))


def quadratic_vote_weight(ip_balance: float, staked_ip: float = 0.0) -> float:
    """Return the quadratic voting weight from available influence points.

    The weight is computed as ``sqrt(ip_balance + staked_ip)``. Any negative
    totals are treated as ``0`` to avoid ``ValueError`` from ``sqrt``.
    """
    return math.sqrt(max(0.0, ip_balance + staked_ip))


class GovernanceQueryService:
    """Narrow read-only governance projection API for orchestration layers."""

    def current_rules(self) -> list[dict[str, Any]]:
        return governance_rules_engine.current_rules()

    def pending_votes(self) -> list[dict[str, Any]]:
        return governance_rules_engine.pending_votes()

    def active_offices(self) -> list[dict[str, Any]]:
        return governance_rules_engine.active_offices()

    def sanctions(self) -> list[dict[str, Any]]:
        return governance_rules_engine.sanctions()

    def read_model(self) -> dict[str, Any]:
        rules = self.current_rules()
        return {
            "rules": rules,
            "current_rules": rules,
            "pending_votes": self.pending_votes(),
            "active_offices": self.active_offices(),
            "sanctions": self.sanctions(),
        }


class GovernanceService:
    """Service coordinating law proposals and voting."""

    def __init__(self) -> None:
        self._knowledge_board: EntryStore | None = None
        self.query = GovernanceQueryService()
        self._step_provider: Callable[[], int] = lambda: 0
        self._transaction_service: KnowledgeGovernanceTransaction | None = None

    def attach_knowledge_board(
        self,
        board: EntryStore | None,
        *,
        step_provider: Callable[[], int] | None = None,
    ) -> None:
        """Attach a board so governance writes become queryable provenance entries."""
        self._knowledge_board = board
        if step_provider is not None:
            self._step_provider = step_provider
        self._transaction_service = (
            KnowledgeGovernanceTransaction(board, step_provider=self._step_provider)
            if board is not None
            else None
        )

    def _current_step(self) -> int:
        try:
            return int(self._step_provider())
        except Exception:
            return 0

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

        if (
            not proposal_entry_id
            or self._knowledge_board is None
            or self._transaction_service is None
        ):
            return approve

        step = self._current_step()
        idempotency_key = make_vote_idempotency_key(
            voter_id=agent.agent_id,
            proposal_entry_id=proposal_entry_id,
            approve=approve,
            step=step,
        )
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
                "idempotency_key": idempotency_key,
            },
        )

        def apply_vote() -> tuple[str, str, bool] | None:
            self._knowledge_board.record_vote(
                voter_agent_id=agent.agent_id,
                proposal_id=proposal_entry_id,
                approve=approve,
            )
            return (agent.agent_id, proposal_entry_id, approve)

        mutation = GovernanceMutation(apply=apply_vote, rollback=lambda _token: None)
        return await self._transaction_service.execute(
            actor_id=agent.agent_id,
            board_entry=vote_entry,
            idempotency_key=idempotency_key,
            governance_mutation=mutation,
            emit_event=_emit_governance_event,
            event_payload={
                "type": "governance_vote",
                "actor_id": agent.agent_id,
                "proposal_entry_id": proposal_entry_id,
                "approve": approve,
            },
        )

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
        step = self._current_step()
        idempotency_key = make_proposal_idempotency_key(
            proposer_id=proposer.agent_id,
            text=text,
            step=step,
        )
        if self._knowledge_board is not None:
            from src.sim.knowledge_board import prepare_entry_payload

            proposal_entry_id, _ = prepare_entry_payload(
                proposal_payload,
                proposer.agent_id,
                self._current_step(),
            )

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

        rule_materialization: dict[str, Any] | None = None
        governance_before = copy.deepcopy(governance_rules_engine._governance_state)
        active_rules_before = copy.deepcopy(governance_rules_engine._active_rules)

        def apply_proposal_mutation() -> dict[str, Any]:
            nonlocal rule_materialization
            rule_materialization = governance_rules_engine.proposal_workflow(
                text,
                proposer_id=proposer.agent_id,
                approved=approved,
                proposal_record=outcome,
            )
            return {
                "governance_state": governance_before,
                "active_rules": active_rules_before,
            }

        def rollback_proposal_mutation(token: Any) -> None:
            if not isinstance(token, dict):
                return
            governance_rules_engine._governance_state = dict(token.get("governance_state", {}))
            governance_rules_engine._active_rules = dict(token.get("active_rules", {}))

        if self._knowledge_board is not None and self._transaction_service is not None:
            proposal_payload.reference_metadata = {
                "idempotency_key": idempotency_key,
                "kind": "proposal",
            }
            proposal_ok = await self._transaction_service.execute(
                actor_id=proposer.agent_id,
                board_entry=proposal_payload,
                idempotency_key=idempotency_key,
                governance_mutation=GovernanceMutation(
                    apply=apply_proposal_mutation,
                    rollback=rollback_proposal_mutation,
                ),
                emit_event=_emit_governance_event,
                event_payload={
                    "type": "governance_proposal",
                    "actor_id": proposer.agent_id,
                    "proposal_text": text,
                },
            )
            if not proposal_ok:
                return False
        else:
            rule_materialization = governance_rules_engine.proposal_workflow(
                text,
                proposer_id=proposer.agent_id,
                approved=approved,
                proposal_record=outcome,
            )

        outcome["rule_materialization"] = rule_materialization or {}
        governance_rule_id = (
            str((rule_materialization or {}).get("rule_id"))
            if isinstance(rule_materialization, dict)
            else None
        )

        if self._knowledge_board is not None:
            for agent, vote in zip(agents, votes):
                await self.vote_weighted(
                    agent,
                    text,
                    weight=1,
                    approve=vote,
                    proposal_entry_id=proposal_entry_id or None,
                    governance_rule_id=governance_rule_id,
                )

            if approved and self._transaction_service is not None:
                law_entry = KnowledgeEntry(
                    content_full=f"Law ratified: {text}",
                    entry_type=KnowledgeEntryType.LAW,
                    parent_entry_id=proposal_entry_id or None,
                    governance_rule_id=governance_rule_id,
                    tags=["governance", "law"],
                    reference_metadata={
                        "relationship": "supersedes",
                        "idempotency_key": f"law:{idempotency_key}",
                    },
                )
                await self._transaction_service.execute(
                    actor_id=proposer.agent_id,
                    board_entry=law_entry,
                    idempotency_key=f"law:{idempotency_key}",
                    governance_mutation=GovernanceMutation(
                        apply=lambda: None,
                        rollback=lambda _token: None,
                    ),
                    emit_event=_emit_governance_event,
                    event_payload={
                        "type": "governance_law_applied",
                        "actor_id": proposer.agent_id,
                        "proposal_text": text,
                    },
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

__all__ = ["GovernanceQueryService", "GovernanceService", "governance", "quadratic_vote_weight"]
