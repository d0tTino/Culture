"""Council Mode orchestration for gathering answers and selecting a winner."""

from __future__ import annotations

import re
import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.agents.council.types import (
    CouncilConfig,
    CouncilMemberConfig,
    CouncilOutcome,
    CouncilQuestion,
    MemberAnswer,
)
from src.infra import llm_client
from src.infra.config import get_config, load_council_config
from src.infra.llm_client import generate_structured_output, generate_text
from src.shared import llm_mocks

DEFAULT_MEMBER_PROMPT = (
    "You are participating in a council of AI personas. Provide a JSON object with keys: "
    "answer (string), reasoning (string), confidence (0-1 float), citations (list of strings)."
)


class MemberResponseModel(BaseModel):
    """Structured response produced by an individual council member."""

    answer: str
    reasoning: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)


class CouncilVoteModel(BaseModel):
    """Structured judgment produced by the council's adjudicator."""

    winning_member_id: str
    scores: dict[str, float] = Field(default_factory=dict)
    summary: str
    reasoning: str


@dataclass(slots=True)
class CouncilContext:
    """Container for derived council settings used during orchestration."""

    config: CouncilConfig
    judge_model: str
    member_model: str


def _slugify(value: str) -> str:
    """Convert display names into a predictable identifier."""

    lowered = value.strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "-", lowered)
    return sanitized.strip("-") or "member"


def _build_council_context() -> CouncilContext:
    """Load the council configuration and normalize member metadata."""

    raw_config = load_council_config()
    default_model = str(get_config("DEFAULT_LLM_MODEL") or "mistral:latest")
    judge_model = str(raw_config.get("judge_model") or default_model)

    members: list[CouncilMemberConfig] = []
    for index, entry in enumerate(raw_config.get("members", [])):
        if not isinstance(entry, Mapping):
            continue
        display_name = str(
            entry.get("display_name")
            or entry.get("name")
            or entry.get("id")
            or f"Member {index + 1}"
        )
        member_id = str(entry.get("id") or _slugify(display_name))
        role = str(entry.get("role") or "Generalist")
        description = str(
            entry.get("description") or f"{display_name} focuses on the {role} perspective."
        )
        system_prompt = str(
            entry.get("system_prompt")
            or f"You are {display_name}, a {role}. Offer concise, grounded answers."
        )
        decision_weight = float(entry.get("decision_weight") or entry.get("weight") or 1.0)
        max_turn_tokens = entry.get("max_turn_tokens")
        metadata = dict(entry.get("metadata") or {})
        model_name = str(entry.get("model") or metadata.get("model") or default_model)
        metadata.setdefault("model", model_name)

        members.append(
            CouncilMemberConfig(
                member_id=member_id,
                display_name=display_name,
                role=role,
                description=description,
                system_prompt=system_prompt,
                decision_weight=decision_weight,
                max_turn_tokens=int(max_turn_tokens) if max_turn_tokens is not None else None,
                metadata=metadata,
            )
        )

    council_config = CouncilConfig(
        enabled=bool(raw_config.get("enabled", True)),
        members=members,
        quorum=raw_config.get("quorum"),
        consensus_threshold=float(raw_config.get("consensus_threshold", 0.67)),
        max_rounds=int(raw_config.get("max_rounds", 1)),
        auto_record_transcript=bool(raw_config.get("auto_record_transcript", True)),
        metadata={"raw_config": raw_config},
    )

    return CouncilContext(config=council_config, judge_model=judge_model, member_model=default_model)


def _format_rag_docs(rag_docs: Sequence[str]) -> str:
    if not rag_docs:
        return "- (no retrieved documents; placeholder RAG list)"
    return "\n".join(f"- {doc}" for doc in rag_docs)


def _build_member_prompt(
    member: CouncilMemberConfig,
    question: CouncilQuestion,
    *,
    extra_context: str | None = None,
    rag_docs: Sequence[str] | None = None,
) -> str:
    rag_section = _format_rag_docs(rag_docs or [])
    additional_context = extra_context or "(no extra context provided)"
    return (
        f"System prompt for {member.display_name} ({member.role}): {member.system_prompt}\n"
        f"Persona description: {member.description}\n\n"
        f"Council question: {question.prompt}\n"
        f"Additional context: {additional_context}\n"
        f"Retrieved documents (RAG):\n{rag_section}\n\n"
        f"{DEFAULT_MEMBER_PROMPT}"
    )


def _ask_council_member(
    member: CouncilMemberConfig,
    question: CouncilQuestion,
    *,
    extra_context: str | None = None,
    rag_docs: Sequence[str] | None = None,
) -> MemberAnswer:
    prompt = _build_member_prompt(member, question, extra_context=extra_context, rag_docs=rag_docs)
    model_name = str((member.metadata or {}).get("model")) if member.metadata else None
    member_model = model_name or "mistral:latest"

    structured = generate_structured_output(
        prompt,
        response_model=MemberResponseModel,
        model=member_model,
        temperature=0.3,
    )

    if structured is None:
        fallback_text = generate_text(prompt, model=member_model, temperature=0.3) or ""
        return MemberAnswer(
            member_id=member.member_id,
            answer=fallback_text,
            reasoning=None,
            confidence=None,
            citations=[],
        )

    return MemberAnswer(
        member_id=member.member_id,
        answer=structured.answer,
        reasoning=structured.reasoning,
        confidence=structured.confidence,
        citations=structured.citations,
    )


def _build_judge_prompt(
    question: CouncilQuestion,
    answers: Sequence[MemberAnswer],
    *,
    extra_context: str | None = None,
    rag_docs: Sequence[str] | None = None,
) -> str:
    rag_section = _format_rag_docs(rag_docs or [])
    additional_context = extra_context or "(no extra context provided)"
    formatted_answers = "\n".join(
        (
            f"Member {answer.member_id}:\n"
            f"Answer: {answer.answer}\n"
            f"Reasoning: {answer.reasoning or '(not provided)'}\n"
        )
        for answer in answers
    )
    return (
        "You are the judging model for council deliberations. Review the provided answers and "
        "select the best response. Provide JSON with keys: winning_member_id (string), scores "
        "(object of member_id to 0-1 float), summary (string), reasoning (string).\n\n"
        f"Council question: {question.prompt}\n"
        f"Additional context: {additional_context}\n"
        f"Retrieved documents (RAG):\n{rag_section}\n\n"
        f"Answers:\n{formatted_answers}\n\n"
        "Return only valid JSON."
    )


def _judge_council_answers(
    context: CouncilContext,
    question: CouncilQuestion,
    answers: Sequence[MemberAnswer],
    *,
    extra_context: str | None = None,
    rag_docs: Sequence[str] | None = None,
) -> CouncilVoteModel | None:
    if not answers:
        return None

    prompt = _build_judge_prompt(question, answers, extra_context=extra_context, rag_docs=rag_docs)
    return generate_structured_output(
        prompt,
        response_model=CouncilVoteModel,
        model=context.judge_model,
        temperature=0.1,
    )


def run_council(
    question: CouncilQuestion,
    *,
    extra_context: str | None = None,
    rag_docs: Sequence[str] | None = None,
) -> CouncilOutcome:
    """Gather answers from council members and select a winner using a judge model."""

    council_context = _build_council_context()
    rag_docs = rag_docs or []

    answers: list[MemberAnswer] = []
    for member in council_context.config.members:
        answers.append(
            _ask_council_member(
                member,
                question,
                extra_context=extra_context,
                rag_docs=rag_docs,
            )
        )

    vote = _judge_council_answers(
        council_context,
        question,
        answers,
        extra_context=extra_context,
        rag_docs=rag_docs,
    )

    winning_member_ids: list[str] = []
    resolution = "No consensus reached."
    summary = None
    metadata: dict[str, Any] | None = None

    if vote is not None:
        winning_member_ids = [vote.winning_member_id]
        metadata = {"scores": vote.scores, "judge_reasoning": vote.reasoning}
        summary = vote.summary
        answer_lookup = {answer.member_id: answer.answer for answer in answers}
        resolution = answer_lookup.get(vote.winning_member_id, "No consensus reached.")

    return CouncilOutcome(
        question=question,
        answers=answers,
        resolution=resolution,
        winning_member_ids=winning_member_ids,
        summary=summary,
        metadata=metadata,
    )


class CouncilOrchestrator:
    """Coordinate concurrent council member calls with optional resource limits."""

    def __init__(self: "CouncilOrchestrator", *, max_concurrency: int = 3) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    @staticmethod
    def _build_member_prompt(member: CouncilMemberConfig, question: CouncilQuestion) -> str:
        return (
            "[council-member-answer] "
            f"member_id={member.member_id} question={question.prompt} context={question.context or ''}"
        )

    @staticmethod
    def _build_judge_prompt(question: CouncilQuestion, answers: Sequence[MemberAnswer]) -> str:
        member_section = " ".join(f"member_id={answer.member_id}" for answer in answers)
        return (
            "[council-judgement] "
            f"question={question.prompt} context={question.context or ''} {member_section}"
        )

    @staticmethod
    def _parse_member_response(member: CouncilMemberConfig, raw: dict[str, Any]) -> MemberAnswer:
        return MemberAnswer(
            member_id=member.member_id,
            answer=str(raw.get("answer", "")),
            reasoning=raw.get("reasoning"),
            confidence=float(raw.get("confidence", 0.0)) if raw.get("confidence") is not None else None,
            citations=list(raw.get("citations", []) or []),
        )

    async def _ask_member_async(
        self: "CouncilOrchestrator",
        member: CouncilMemberConfig,
        question: CouncilQuestion,
    ) -> MemberAnswer:
        prompt = self._build_member_prompt(member, question)
        async with self._semaphore:
            response = await asyncio.to_thread(llm_client.client.generate, prompt=prompt)
        payload = json.loads(str(response.get("response", "{}")))
        return self._parse_member_response(member, payload)

    async def _judge_answers_async(
        self: "CouncilOrchestrator",
        question: CouncilQuestion,
        answers: Sequence[MemberAnswer],
    ) -> CouncilOutcome:
        prompt = self._build_judge_prompt(question, answers)
        response = await asyncio.to_thread(llm_client.client.generate, prompt=prompt)
        payload = json.loads(str(response.get("response", "{}")))

        winner = str(payload.get("winner") or "")
        votes = payload.get("votes") or {}
        metrics = payload.get("metrics") or {}

        return CouncilOutcome(
            question=question,
            answers=answers,
            resolution=str(payload.get("resolution") or "No consensus reached."),
            winning_member_ids=[winner] if winner else [],
            summary=payload.get("summary"),
            metadata={"votes": votes, "metrics": metrics},
        )

    async def _gather_answers(
        self: "CouncilOrchestrator",
        config: CouncilConfig,
        question: CouncilQuestion,
    ) -> tuple[list[MemberAnswer], bool]:
        tasks = {
            asyncio.create_task(self._ask_member_async(member, question)): index
            for index, member in enumerate(config.members)
        }
        answers: list[tuple[int, MemberAnswer]] = []
        budget_exhausted = False
        try:
            pending_tasks = set(tasks.keys())
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    try:
                        answer = task.result()
                    except RuntimeError as exc:
                        if "budget" in str(exc).lower():
                            budget_exhausted = True
                            continue
                        raise
                    else:
                        answers.append((tasks[task], answer))
                if budget_exhausted:
                    pending_tasks = set()
                    break
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        ordered_answers = [
            answer for _, answer in sorted(answers, key=lambda pair: pair[0])
        ]
        return ordered_answers, budget_exhausted

    async def _deliberate_async(
        self: "CouncilOrchestrator",
        config: CouncilConfig,
        question: CouncilQuestion,
    ) -> CouncilOutcome:
        answers, budget_exhausted = await self._gather_answers(config, question)

        if budget_exhausted:
            return CouncilOutcome(
                question=question,
                answers=answers,
                resolution="Insufficient DU budget; partial council outcome",
                winning_member_ids=[],
                summary=None,
                metadata={
                    "du_exhausted": True,
                    "partial": True,
                    "completed_members": [a.member_id for a in answers],
                },
            )

        if not answers:
            return CouncilOutcome(
                question=question,
                answers=[],
                resolution="No answers produced.",
                winning_member_ids=[],
                summary=None,
                metadata=None,
            )

        return await self._judge_answers_async(question, answers)

    def deliberate(
        self: "CouncilOrchestrator",
        config: CouncilConfig,
        question: CouncilQuestion,
    ) -> CouncilOutcome:
        """Synchronously orchestrate the council deliberation."""

        return asyncio.run(self._deliberate_async(config, question))
