from __future__ import annotations

from collections.abc import Iterable

from src.agents.core.base_agent import Agent

from .service import governance


async def _vote(agent: Agent, proposal: str) -> bool:
    """Proxy to :class:`GovernanceService.vote`."""
    return await governance.vote(agent, proposal)


async def propose_law(proposer: Agent, text: str, agents: Iterable[Agent]) -> bool:
    """Delegate to :class:`GovernanceService`."""
    return await governance.propose_law(proposer, text, agents)
