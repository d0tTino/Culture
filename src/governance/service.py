from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable
from typing import cast

from typing_extensions import Self

from src.agents.core.base_agent import Agent
from src.infra.ledger import ledger
from src.utils.policy import evaluate_with_opa

from .law_board import law_board


def quadratic_vote_weight(ip_balance: float, staked_ip: float = 0.0) -> float:
    """Return the quadratic voting weight from available influence points.

    The weight is computed as ``sqrt(ip_balance + staked_ip)``. Any negative
    totals are treated as ``0`` to avoid ``ValueError`` from ``sqrt``.
    """
    return math.sqrt(max(0.0, ip_balance + staked_ip))


class GovernanceService:
    """Service coordinating law proposals and voting."""

    async def vote(self: Self, agent: Agent, proposal: str) -> bool:
        """Return the agent's vote (True=approve) using the OPA policy."""
        allowed, _ = await evaluate_with_opa(proposal)
        return allowed

    async def propose_law(
        self: Self,
        proposer: Agent,
        text: str,
        agents: Iterable[Agent],
        vote_weights: dict[str, int] | None = None,
    ) -> bool:
        """Propose ``text`` to ``agents`` and persist the result.

        When ``vote_weights`` is provided, each agent may cast multiple votes.
        The cost in influence points (IP) for casting ``n`` votes is ``n^2``.
        ``ip_spent`` records the total IP deducted for this proposal.
        """
        allowed, _ = await evaluate_with_opa(text)
        if not allowed:
            return False

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
            start_balances = await asyncio.gather(
                *[ledger.get_balance_async(a.agent_id) for a in agents],
                return_exceptions=True,
            )
            spend_tasks = []
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
        try:
            ledger.record_law_proposal(
                proposer.agent_id,
                text,
                approved,
                yes_weight,
                no_weight,
                ip_spent,
            )
        except Exception:
            pass
        return approved

    def get_proposals(self: Self, limit: int | None = None) -> list[dict[str, object]]:
        """Return stored law proposals."""
        try:
            return ledger.get_law_proposals(limit)
        except Exception:
            return []


governance = GovernanceService()

__all__ = ["GovernanceService", "governance", "quadratic_vote_weight"]
