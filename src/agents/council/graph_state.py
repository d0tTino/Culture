"""State and helpers for council-focused LangGraph flows."""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired

from src.agents.council.orchestrator import run_council
from src.agents.council.types import CouncilOutcome, CouncilQuestion

logger = logging.getLogger(__name__)


class CouncilState(TypedDict):
    """State shared between council graph nodes."""

    question: str
    messages: list[str]
    council_outcome: NotRequired[CouncilOutcome | None]
    final_answer: NotRequired[str | None]


class CouncilStateModel(BaseModel):
    """Pydantic representation of :class:`CouncilState`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    messages: list[str] = Field(default_factory=list)
    council_outcome: CouncilOutcome | None = None
    final_answer: str | None = None


async def council_outcome_node(state: CouncilState) -> dict[str, CouncilOutcome | str | None]:
    """Resolve a council question and store the result in the graph state."""

    rag_docs = [str(message) for message in state.get("messages", [])]
    question = CouncilQuestion(
        question_id="council-question",
        question=state.get("question", ""),
        context="\n".join(rag_docs) if rag_docs else None,
        rag_documents=rag_docs,
    )

    try:
        outcome: CouncilOutcome = await asyncio.to_thread(
            run_council, question, extra_context=question.extra_context, rag_docs=rag_docs
        )
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.error("Council deliberation failed: %s", exc, exc_info=True)
        return {"council_outcome": None, "final_answer": None}

    final_answer = outcome.resolution or outcome.summary or ""
    return {"council_outcome": outcome, "final_answer": final_answer}
