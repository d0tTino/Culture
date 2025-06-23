from __future__ import annotations

import asyncio
from collections.abc import Iterable

from src.agents.core.base_agent import Agent
from src.utils.policy import evaluate_with_opa


async def _vote(agent: Agent, proposal: str) -> bool:
    """Simple voting behavior: agent votes according to OPA evaluation."""
    allowed, _ = await evaluate_with_opa(proposal)
    return allowed


async def propose_law(proposer: Agent, text: str, agents: Iterable[Agent]) -> bool:
    """Propose a law and collect votes from all agents."""
    allowed, _ = await evaluate_with_opa(text)
    if not allowed:
        return False

    votes = await asyncio.gather(*[_vote(a, text) for a in agents])
    yes_votes = sum(1 for v in votes if v)
    return yes_votes > len(votes) / 2
