"""Lightweight orchestration utilities for Council Mode simulations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from src.agents.council.types import (
    CouncilConfig,
    CouncilOutcome,
    CouncilQuestion,
    MemberAnswer,
)
from src.infra import llm_client


class CouncilJudgeResponse(BaseModel):
    """Structured output produced by the judge step."""

    winner: str = Field(description="Member ID that produced the winning answer")
    votes: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    resolution: str
    summary: str | None = None


class CouncilMemberResponse(BaseModel):
    """Structured output produced for each council member."""

    answer: str
    reasoning: str
    confidence: float | None = None
    citations: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class CouncilOrchestrator:
    """Coordinate member calls and adjudication for Council Mode."""

    model: str = "mistral:latest"

    def _member_prompt(self, question: CouncilQuestion, member_id: str) -> str:
        return (
            "[council-member-answer]\n"
            f"member_id={member_id}\n"
            f"question={question.prompt}\n"
            f"context={question.context or ''}"
        )

    def _judge_prompt(
        self, question: CouncilQuestion, answers: Sequence[MemberAnswer]
    ) -> str:
        rendered_answers = "\n".join(
            f"member_id={answer.member_id} :: answer={answer.answer}"
            for answer in answers
        )
        return (
            "[council-judgement]\n"
            f"question={question.prompt}\n"
            f"context={question.context or ''}\n"
            f"answers=\n{rendered_answers}"
        )

    def _parse_member_response(
        self, payload: dict[str, object], member_id: str
    ) -> MemberAnswer:
        parsed = CouncilMemberResponse(**payload)
        return MemberAnswer(
            member_id=member_id,
            answer=parsed.answer,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            citations=list(parsed.citations),
        )

    def _collect_member_answers(
        self, question: CouncilQuestion, members: Iterable[str]
    ) -> list[MemberAnswer]:
        answers: list[MemberAnswer] = []
        for member_id in members:
            prompt = self._member_prompt(question, member_id)
            raw_response = llm_client.client.generate(prompt=prompt, model=self.model)
            payload = json.loads(str(raw_response.get("response", "{}")))
            answers.append(self._parse_member_response(payload, member_id))
        return answers

    def _adjudicate(self, question: CouncilQuestion, answers: Sequence[MemberAnswer]) -> CouncilJudgeResponse:
        prompt = self._judge_prompt(question, answers)
        raw_response = llm_client.client.generate(prompt=prompt, model=self.model)
        payload = json.loads(str(raw_response.get("response", "{}")))
        return CouncilJudgeResponse(**payload)

    def deliberate(self, config: CouncilConfig, question: CouncilQuestion) -> CouncilOutcome:
        if not config.enabled:
            raise ValueError("Council is disabled in the provided configuration")

        active_members = [member.member_id for member in config.members if member.decision_weight > 0]
        member_answers = self._collect_member_answers(question, active_members)
        judgement = self._adjudicate(question, member_answers)

        metadata = {"votes": judgement.votes, "metrics": judgement.metrics}
        return CouncilOutcome(
            question=question,
            answers=member_answers,
            resolution=judgement.resolution,
            winning_member_ids=[judgement.winner],
            summary=judgement.summary,
            metadata=metadata,
        )

