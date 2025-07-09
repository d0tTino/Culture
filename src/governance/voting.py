from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable

from src.agents.core.base_agent import Agent
from src.infra.ledger import ledger
from src.utils.policy import evaluate_with_opa

from .law_board import law_board


async def _vote(agent: Agent, proposal: str) -> bool:
    """Simple voting behavior: agent votes according to OPA evaluation."""
    allowed, _ = await evaluate_with_opa(proposal)
    return allowed


async def propose_law(proposer: Agent, text: str, agents: Iterable[Agent]) -> bool:
    """Propose a law and collect weighted votes from all agents."""
    allowed, _ = await evaluate_with_opa(text)
    if not allowed:
        return False

    votes = await asyncio.gather(*[_vote(a, text) for a in agents])
    weights = []
    for a in agents:
        base_ip = getattr(a.state, "ip", 0.0)
        try:
            staked = ledger.get_staked_ip(a.agent_id)
        except Exception:
            staked = 0.0
        weights.append(math.sqrt(base_ip + staked))
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
        )
    except Exception:
        pass
    return approved
