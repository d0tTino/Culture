"""LangGraph node wrapping ``MultiLayerRetriever.retrieve``."""

from __future__ import annotations

from typing import Any, cast

import tiktoken
from opentelemetry import trace

from src.agents.memory.multi_layer_retriever import MultiLayerRetriever

from .basic_agent_types import AgentTurnState

tracer = trace.get_tracer(__name__)


async def retriever_node(state: AgentTurnState) -> dict[str, Any]:
    """Retrieve memories while enforcing a per-call token budget.

    The state is expected to provide a ``memory_retriever`` key containing a
    ``MultiLayerRetriever`` instance. Results are pruned so that the total
    token count does not exceed ``state['token_budget']`` (when provided).
    If pruning occurs a ``memory.token_budget_exceeded`` span is emitted.
    """

    retriever = cast(MultiLayerRetriever | None, state.get("memory_retriever"))
    if retriever is None:
        return {"memory_context": [], "memory_history_list": []}

    agent_id = cast(str, state.get("agent_id", ""))
    query = cast(str, state.get("query", ""))
    k = cast(int, state.get("k", 5))

    results = await retriever.retrieve(agent_id, query, k)

    token_budget = cast(int | None, state.get("token_budget"))
    if token_budget is not None:
        tokenizer = retriever.tokenizer or tiktoken.get_encoding("cl100k_base")
        tokens = 0
        limited: list[dict[str, Any]] = []
        for mem in results:
            text = str(mem.get("content", ""))
            t = len(tokenizer.encode(text))
            if tokens + t > token_budget:
                with tracer.start_as_current_span("memory.token_budget_exceeded") as span:
                    span.set_attribute("memory.agent_id", agent_id)
                    span.set_attribute("memory.token_budget", token_budget)
                break
            limited.append(mem)
            tokens += t
        results = limited

    return {
        "memory_history_list": results,
        "memory_context": [str(m.get("content", "")) for m in results],
    }
