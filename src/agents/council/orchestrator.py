"""Council Mode orchestration for gathering answers and selecting a winner."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.shared import llm_mocks
from src.agents.council.types import (
    CouncilConfig,
    CouncilMemberConfig,
    CouncilOutcome,
    CouncilQuestion,
    MemberAnswer,
)
from src.agents.council.fitness_store import CouncilFitnessStore, council_fitness_store
from src.agents.memory.multi_layer_retriever import MultiLayerRetriever
from src.infra import llm_client
from src.infra.config import get_config, load_council_config
from src.infra import llm_client
from src.infra.llm_client import (
    generate_structured_output,
    generate_text,
    is_mock_mode_enabled,
)
from src.sim.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

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

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    winning_member_id: str = Field(
        alias="winning_member_id", validation_alias="winner"
    )
    scores: dict[str, float] = Field(default_factory=dict, alias="votes")
    metrics: dict[str, float] = Field(default_factory=dict)
    summary: str
    reasoning: str | None = None
    resolution: str | None = None


@dataclass(slots=True)
class CouncilContext:
    """Container for derived council settings used during orchestration."""

    config: CouncilConfig
    judge_model: str
    member_model: str


def _stringify_memory_doc(doc: Any) -> str:
    if isinstance(doc, str):
        return doc

    if isinstance(doc, Mapping):
        content = doc.get("content") or doc.get("text") or ""
        metadata = doc.get("metadata")
        if isinstance(metadata, Mapping):
            source = metadata.get("source") or metadata.get("id") or metadata.get("memory_id")
            if source:
                if content:
                    return f"{content} (source: {source})"
                return str(source)
        if content:
            return str(content)

    try:
        return json.dumps(doc)
    except Exception:  # pragma: no cover - defensive
        return str(doc)


def _resolve_question_agent_id(question: CouncilQuestion, explicit: str | None) -> str | None:
    if explicit:
        return explicit

    metadata = question.metadata or {}
    if not isinstance(metadata, Mapping):
        return None

    for key in ("agent_id", "originator_id", "requester_id", "author_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


async def _populate_question_rag_documents(
    question: CouncilQuestion,
    *,
    base_documents: Sequence[str] | None = None,
    memory_service: Any | None = None,
    memory_retriever: MultiLayerRetriever | None = None,
    agent_id: str | None = None,
    top_k: int = 5,
    token_budget: int | None = None,
) -> Sequence[str]:
    question.rag_documents = list(question.rag_documents or [])
    if base_documents:
        question.rag_documents.extend(str(doc) for doc in base_documents)

    retriever = memory_retriever or getattr(memory_service, "retriever", None)
    if retriever is None:
        return question.rag_documents

    agent_identifier = _resolve_question_agent_id(question, agent_id)
    if not agent_identifier:
        logger.debug("Council question missing agent identifier; skipping RAG retrieval")
        return question.rag_documents

    query = question.prompt
    if question.context:
        query = f"{question.prompt}\n\n{question.context}"

    try:
        results = await retriever.retrieve(
            agent_identifier, query, k=top_k, token_budget=token_budget
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to retrieve council RAG documents: %s", exc, exc_info=True)
        return question.rag_documents

    question.rag_documents.extend(_stringify_memory_doc(doc) for doc in results or [])
    return question.rag_documents


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
    max_concurrent_calls = raw_config.get("max_concurrent_calls")
    if max_concurrent_calls is None:
        max_concurrent_calls = get_config("COUNCIL_MAX_CONCURRENT_CALLS")
    du_budget_per_question = raw_config.get("du_budget_per_question")
    if du_budget_per_question is None:
        du_budget_per_question = get_config("DU_BUDGET_PER_QUESTION")

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
        max_concurrent_calls=int(max_concurrent_calls) if max_concurrent_calls else None,
        du_budget_per_question=(
            float(du_budget_per_question) if du_budget_per_question is not None else None
        ),
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
        "[council-member-answer] "
        f"member_id={member.member_id} question={question.prompt} context={question.context or ''} "
        f"extra_context={additional_context} rag_docs={rag_section}\n"
        f"System prompt for {member.display_name} ({member.role}): {member.system_prompt}\n"
        f"Persona description: {member.description}\n\n"
        f"{DEFAULT_MEMBER_PROMPT}"
    )


def _ask_council_member(
    member: CouncilMemberConfig,
    question: CouncilQuestion,
    *,
    extra_context: str | None = None,
    rag_docs: Sequence[str] | None = None,
    agent_state: Any | None = None,
) -> MemberAnswer:
    prompt = _build_member_prompt(member, question, extra_context=extra_context, rag_docs=rag_docs)
    model_name = str((member.metadata or {}).get("model")) if member.metadata else None
    member_model = model_name or "mistral:latest"
    track_mock_usage = is_mock_mode_enabled()
    if track_mock_usage:
        response = llm_client.client.generate(prompt=prompt)
        payload = json.loads(str(response.get("response", "{}")))
        structured = MemberResponseModel.model_validate(payload)
    else:
        structured = generate_structured_output(
            prompt,
            response_model=MemberResponseModel,
            model=member_model,
            temperature=0.3,
            agent_state=agent_state,
        )
        llm_client.client.generate(prompt=telemetry_prompt)

    structured = generate_structured_output(
        prompt,
        response_model=MemberResponseModel,
        model=member_model,
        temperature=0.3,
        agent_state=agent_state,
    )

    if structured is None:
        fallback_text = (
            generate_text(
                prompt, model=member_model, temperature=0.3, agent_state=agent_state
            )
            or ""
        )
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
    member_section = " ".join(f"member_id={answer.member_id}" for answer in answers)
    return (
        "[council-judgement] "
        f"question={question.prompt} context={question.context or ''} "
        f"extra_context={additional_context} rag_docs={rag_section} {member_section}"
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

    track_mock_usage = is_mock_mode_enabled()
    prompt = _build_judge_prompt(question, answers, extra_context=extra_context, rag_docs=rag_docs)
    if track_mock_usage:
        response = llm_client.client.generate(prompt=prompt)
        payload = json.loads(str(response.get("response", "{}")))
        return CouncilVoteModel.model_validate(payload)

    return generate_structured_output(
        prompt,
        response_model=CouncilVoteModel,
        model=context.judge_model,
        temperature=0.1,
    )


class CouncilOrchestrator:
    """Coordinate concurrent council member calls with DU budgeting."""

    def __init__(
        self,
        *,
        max_concurrent_calls: int | None = None,
        max_concurrency: int | None = None,
        du_budget_per_question: float | None = None,
        memory_service: Any | None = None,
        memory_retriever: MultiLayerRetriever | None = None,
        rag_top_k: int | None = None,
        rag_token_limit: int | None = None,
        fitness_store: CouncilFitnessStore | None = None,
    ) -> None:
        default_concurrency = int(get_config("COUNCIL_MAX_CONCURRENT_CALLS") or 1)
        resolved_concurrency = max_concurrent_calls
        if resolved_concurrency is None and max_concurrency is not None:
            resolved_concurrency = max_concurrency
        self.max_concurrent_calls = int(resolved_concurrency or default_concurrency)
        self.max_concurrency = self.max_concurrent_calls
        default_budget = float(get_config("DU_BUDGET_PER_QUESTION") or 0.0)
        self.du_budget_per_question = float(
            default_budget if du_budget_per_question is None else du_budget_per_question
        )
        self.memory_service = memory_service
        self.memory_retriever = memory_retriever
        self.rag_top_k = int(rag_top_k or get_config("MEMORY_RETRIEVER_TOP_K") or 5)
        token_limit_value = rag_token_limit
        if token_limit_value is None:
            token_limit_value = get_config("MEMORY_RETRIEVER_TOKEN_LIMIT")
        self.rag_token_limit = int(token_limit_value) if token_limit_value else None
        self.fitness_store = fitness_store or council_fitness_store

    def _resolve_context(self, config: CouncilConfig | None = None) -> CouncilContext:
        base_context = _build_council_context()
        if config is None:
            return base_context
        return CouncilContext(
            config=config,
            judge_model=base_context.judge_model,
            member_model=base_context.member_model,
        )

    def deliberate(
        self,
        config: CouncilConfig | None,
        question: CouncilQuestion,
        *,
        extra_context: str | None = None,
        rag_docs: Sequence[str] | None = None,
    ) -> CouncilOutcome:
        """Synchronously deliberate by awaiting the async implementation."""

        context = self._resolve_context(config)
        return asyncio.run(
            self.adeliberate(context, question, extra_context=extra_context, rag_docs=rag_docs)
        )

    async def adeliberate(
        self,
        context: CouncilContext,
        question: CouncilQuestion,
        *,
        extra_context: str | None = None,
        rag_docs: Sequence[str] | None = None,
    ) -> CouncilOutcome:
        rag_docs = await _populate_question_rag_documents(
            question,
            base_documents=rag_docs,
            memory_service=self.memory_service,
            memory_retriever=self.memory_retriever,
            top_k=self.rag_top_k,
            token_budget=self.rag_token_limit,
        )
        metrics: dict[str, Any] = {
            "du_budget_exhausted": False,
            "du_budget_per_member": self._resolve_du_budget(context.config),
        }
        member_states = self._allocate_du_budgets(context.config, metrics)

        answers = await self._gather_member_answers(
            context,
            question,
            member_states,
            extra_context=extra_context,
            rag_docs=rag_docs,
            metrics=metrics,
        )

        if metrics.get("du_budget_exhausted"):
            outcome = CouncilOutcome(
                question=question,
                answers=answers,
                resolution="Insufficient DU budget; partial council outcome",
                winning_member_ids=[],
                summary=None,
                metadata={
                    "du_exhausted": True,
                    "partial": True,
                    "completed_members": [answer.member_id for answer in answers],
                },
            )
            await self._record_outcome_metrics(outcome)
            return outcome

        vote = None
        if answers:
            vote = await asyncio.to_thread(
                _judge_council_answers,
                context,
                question,
                answers,
                extra_context=extra_context,
                rag_docs=rag_docs,
            )

        base_metrics = dict(metrics or {})
        base_metrics.setdefault("du_budget_exhausted", False)
        base_metrics.setdefault(
            "du_budget_per_member", self._resolve_du_budget(context.config)
        )

        return self._build_outcome(question, answers, vote, base_metrics)

    def _resolve_du_budget(self, config: CouncilConfig) -> float:
        if config.du_budget_per_question is not None:
            return float(config.du_budget_per_question)
        return float(self.du_budget_per_question)

    def _allocate_du_budgets(
        self, config: CouncilConfig, metrics: dict[str, Any]
    ) -> dict[str, SimpleNamespace]:
        budget = self._resolve_du_budget(config)
        member_states: dict[str, SimpleNamespace] = {}
        try:
            resource_manager = get_resource_manager()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Resource manager unavailable for council DU budgeting: %s", exc)
            resource_manager = None

        for member in config.members:
            state = SimpleNamespace(agent_id=member.member_id, du=budget, ip=0.0)
            member_states[member.member_id] = state
            if resource_manager is None:
                continue
            try:
                resource_manager.set_du_budget(member.member_id, budget)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to allocate DU budget for %s: %s", member.member_id, exc)

        metrics["du_budget_per_member"] = budget
        return member_states

    async def _gather_member_answers(
        self,
        context: CouncilContext,
        question: CouncilQuestion,
        member_states: Mapping[str, SimpleNamespace],
        *,
        extra_context: str | None,
        rag_docs: Sequence[str],
        metrics: dict[str, Any],
    ) -> list[MemberAnswer]:
        semaphore = asyncio.Semaphore(
            max(1, context.config.max_concurrent_calls or self.max_concurrent_calls)
        )

        async def _call_member(member: CouncilMemberConfig) -> MemberAnswer | None:
            state = member_states.get(member.member_id)
            try:
                async with semaphore:
                    return await asyncio.to_thread(
                        _ask_council_member,
                        member,
                        question,
                        extra_context=extra_context,
                        rag_docs=rag_docs,
                        agent_state=state,
                    )
            except RuntimeError as exc:
                if "budget" in str(exc).lower():
                    metrics["du_budget_exhausted"] = True
                    logger.warning(
                        "DU budget exhausted for member %s: %s", member.member_id, exc
                    )
                else:
                    logger.exception("Council member call failed: %s", exc)
                return None

        results = await asyncio.gather(
            *[_call_member(member) for member in context.config.members],
            return_exceptions=True,
        )

        answers: list[MemberAnswer] = []
        for result in results:
            if isinstance(result, Exception):
                if "budget" in str(result).lower():
                    metrics["du_budget_exhausted"] = True
                else:
                    logger.exception("Unexpected error during council call", exc_info=result)
                continue
            if result is not None:
                answers.append(result)
        return answers

    def _build_outcome(
        self,
        question: CouncilQuestion,
        answers: list[MemberAnswer],
        vote: CouncilVoteModel | None,
        metrics: Mapping[str, Any],
    ) -> CouncilOutcome:
        winning_member_ids: list[str] = []
        resolution = "No consensus reached."
        summary = None
        metadata: dict[str, Any] = {"metrics": dict(metrics)}
        fitness_snapshot: Mapping[str, Any] | None = None

        if vote is not None:
            winning_member_ids = [vote.winning_member_id]
            metadata.update({"scores": vote.scores, "judge_reasoning": vote.reasoning})
            metadata.get("metrics", {}).update(vote.metrics)
            summary = vote.summary
            answer_lookup = {answer.member_id: answer.answer for answer in answers}
            resolution = vote.resolution or answer_lookup.get(
                vote.winning_member_id, resolution
            )
            fitness_snapshot = self.fitness_store.update_from_vote(
                question, answers, vote
            )

        if fitness_snapshot is not None:
            metadata["fitness"] = fitness_snapshot

        return CouncilOutcome(
            question=question,
            answers=answers,
            resolution=resolution,
            winning_member_ids=winning_member_ids,
            summary=summary,
            metadata=metadata,
        )

    def serialize_metrics(self) -> dict[str, list[dict[str, float]]]:
        """Return a snapshot of aggregated council metrics."""

        return council_stats_store.serialize_metrics()

    async def serialize_metrics_async(self) -> dict[str, list[dict[str, float]]]:
        return await council_stats_store.serialize_metrics_async()


def run_council(
    question: CouncilQuestion,
    *,
    extra_context: str | None = None,
    rag_docs: Sequence[str] | None = None,
) -> CouncilOutcome:
    """Gather answers from council members and select a winner using a judge model."""

    orchestrator = CouncilOrchestrator()
    return orchestrator.deliberate(
        _build_council_context().config,
        question,
        extra_context=extra_context,
        rag_docs=rag_docs,
    )
