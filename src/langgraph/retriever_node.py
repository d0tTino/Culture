"""LangGraph node for hybrid episodic/semantic memory retrieval."""

from __future__ import annotations

from typing import Any

from typing_extensions import Self


class RetrieverNode:
    """Retrieve memories while enforcing a per-turn token budget.

    The provided ``memory_service`` is expected to merge episodic and semantic
    memories. Retrieved items are truncated so that the combined number of
    whitespace-separated tokens does not exceed ``token_cap``.
    """

    def __init__(self: Self, memory_service: Any, k: int = 5, token_cap: int = 1000) -> None:
        self.memory_service = memory_service
        self.k = k
        self.token_cap = token_cap

    async def __call__(self: Self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve relevant memories for the given state."""
        agent_id = str(state.get("agent_id", ""))
        query = str(state.get("query", ""))
        memories = await self.memory_service.retrieve_relevant_memories(agent_id, query, self.k)
        limited: list[dict[str, Any]] = []
        tokens = 0
        for mem in memories:
            text = mem.get("content", "")
            tokens += len(text.split())
            if tokens > self.token_cap:
                break
            limited.append(mem)
        state["memories"] = limited
        return state
