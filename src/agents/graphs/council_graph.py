"""LangGraph wrapper for running council deliberations."""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict, cast

from langgraph.graph import END, StateGraph
from src.agents.council.orchestrator import run_council
from src.agents.council.types import CouncilOutcome, CouncilQuestion


class CouncilGraphState(TypedDict, total=False):
    """State tracked through the council graph."""

    question: str
    question_id: str
    context: dict[str, Any] | str | None
    rag_documents: list[str]
    council_outcome: CouncilOutcome
    final_answer: str


async def council_node(state: CouncilGraphState) -> dict[str, Any]:
    """Run the council and persist its outcome in the graph state."""

    question = CouncilQuestion(
        question_id=cast(str, state.get("question_id", "council-question")),
        question=cast(str, state.get("question", "")),
        context=state.get("context"),
        rag_documents=state.get("rag_documents", []),
    )

    outcome = await asyncio.to_thread(
        run_council,
        question,
        extra_context=question.extra_context,
        rag_docs=state.get("rag_documents"),
    )

    final_answer = outcome.summary or outcome.resolution
    return {"council_outcome": outcome, "final_answer": final_answer}


def build_graph() -> Any:
    """Compile the council graph and return its executor."""

    graph_builder = StateGraph(CouncilGraphState)
    graph_builder.add_node("council", council_node)
    graph_builder.set_entry_point("council")
    graph_builder.add_edge("council", END)
    return graph_builder.compile()


__all__ = ["CouncilGraphState", "build_graph", "council_node"]
