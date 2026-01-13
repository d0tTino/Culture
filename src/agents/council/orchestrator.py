"""Council Mode orchestration for gathering answers and selecting a winner."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from src.agents.council.fitness_store import CouncilFitnessStore, council_fitness_store
from src.agents.council.stats_store import council_stats_store
from src.agents.council.types import (
    CouncilConfig,
    CouncilMemberConfig,
    CouncilOutcome,
    CouncilQuestion,
    MemberAnswer,
)
from src.agents.memory.multi_layer_retriever import MultiLayerRetriever
from src.infra import llm_client
from src.infra.config import get_config, load_council_config
from src.infra.llm_client import (
    generate_structured_output,
    generate_text,
    is_mock_mode_enabled,
)
from src.sim.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)
LLM_MAX_ATTEMPTS = 2
JUDGE_SCORE_CATEGORIES = ("correctness", "clarity", "usefulness", "safety")

DEFAULT_MEMBER_PROMPT = (
    "You are participating in a council of AI personas. Stay fully in character and keep your unique voice. "
    "Provide a JSON object with keys: answer (string), reasoning (string), confidence (0-1 float), citations (list of strings)."
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
        validation_alias=AliasChoices("winner_id", "winner", "winning_member_id")
    )
    votes: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    summary: str
    reasoning: str | None = None
    resolution: str | None = None


class CouncilPeerVoteModel(BaseModel):
    """Structured peer vote produced by an individual council member."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    winner_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("winner_id", "preferred_member_id", "preferred_member"),
    )
    votes: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    reasoning: str | None = None


@dataclass(slots=True)
class CouncilContext:
    """Container for derived council settings used during orchestration."""

    config: CouncilConfig
    judge_model: str
    member_model: str
    allow_remote_models: bool


class CouncilRunMetrics:
    """Record council metrics while tolerating disabled metric backends."""

    def __init__(self) -> None:
        try:  # pragma: no cover - optional dependency
            from src.interfaces import metrics as prom_metrics
        except Exception:  # pragma: no cover - defensive fallback
            prom_metrics = None

        self._metrics = prom_metrics
        self.enabled = prom_metrics is not None
        self._runs_total = getattr(prom_metrics, "COUNCIL_RUNS_TOTAL", None)
        self._member_wins_total = getattr(prom_metrics, "COUNCIL_MEMBER_WINS_TOTAL", None)
        self._member_score = getattr(prom_metrics, "COUNCIL_MEMBER_SCORE", None)
        self._pairwise_agreement = getattr(
            prom_metrics, "COUNCIL_PAIRWISE_AGREEMENT", None
        )
        self._collusion_warnings_total = getattr(
            prom_metrics, "COUNCIL_COLLUSION_WARNINGS_TOTAL", None
        )
        self._du_budget = getattr(prom_metrics, "COUNCIL_DU_BUDGET", None)
        self._du_spend = getattr(prom_metrics, "COUNCIL_DU_SPEND", None)
        self._latency_ms = getattr(prom_metrics, "COUNCIL_LATENCY_MS", None)

    def _increment(
        self, counter: Any | None, *, labels: Mapping[str, str] | None = None, amount: int = 1
    ) -> None:
        if not self.enabled or counter is None:
            return
        try:
            if labels and hasattr(counter, "labels"):
                counter.labels(**dict(labels)).inc(amount)
            elif hasattr(counter, "inc"):
                counter.inc(amount)
        except Exception:
            return

    def _set_gauge(
        self, gauge: Any | None, value: float, *, labels: Mapping[str, str] | None = None
    ) -> None:
        if not self.enabled or gauge is None:
            return
        try:
            if labels and hasattr(gauge, "labels"):
                gauge.labels(**dict(labels)).set(value)
            elif hasattr(gauge, "set"):
                gauge.set(value)
        except Exception:
            return

    def record_run(self) -> None:
        self._increment(self._runs_total)

    def record_latency(self, stage: str, duration_ms: float) -> None:
        self._set_gauge(self._latency_ms, duration_ms, labels={"stage": stage})

    def record_member_wins(self, member_ids: Sequence[str]) -> None:
        for member_id in member_ids:
            if member_id:
                self._increment(
                    self._member_wins_total, labels={"member_id": str(member_id)}, amount=1
                )

    def record_member_scores_from_vote(self, vote: CouncilVoteModel | None) -> None:
        if not vote:
            return
        member_scores = vote.metrics.get("member_scores") if vote.metrics else None
        if isinstance(member_scores, Mapping):
            for member_id, categories in member_scores.items():
                if not isinstance(categories, Mapping):
                    continue
                for category, score in categories.items():
                    if isinstance(score, (int, float)):
                        self._set_gauge(
                            self._member_score,
                            float(score),
                            labels={"member_id": str(member_id), "category": str(category)},
                        )
        self.record_member_wins([vote.winning_member_id])

    def record_fitness_snapshot(self, snapshot: Mapping[str, Any] | None) -> None:
        if not snapshot:
            return
        members = snapshot.get("members")
        if not isinstance(members, Mapping):
            return
        for member_id, stats in members.items():
            if not isinstance(stats, Mapping):
                continue
            for category in ("wins", "participations", "win_rate", "agreement_score"):
                value = stats.get(category)
                if isinstance(value, (int, float)):
                    self._set_gauge(
                        self._member_score,
                        float(value),
                        labels={"member_id": str(member_id), "category": category},
                    )

        pairs = snapshot.get("pairs")
        if isinstance(pairs, Mapping):
            self.record_pairwise_agreements(pairs)

        warnings = snapshot.get("warnings")
        if isinstance(warnings, Sequence):
            self.record_collusion_warnings(warnings)

    def record_du_usage(
        self, member_states: Mapping[str, SimpleNamespace], budget: float | None
    ) -> None:
        if budget is None:
            return
        try:
            budget_value = float(budget)
        except (TypeError, ValueError):
            return
        for member_id, state in member_states.items():
            remaining = getattr(state, "du", None)
            if remaining is None:
                continue
            try:
                spent = max(budget_value - float(remaining), 0.0)
            except (TypeError, ValueError):
                continue
            self._set_gauge(self._du_budget, budget_value, labels={"member_id": member_id})
            self._set_gauge(self._du_spend, spent, labels={"member_id": member_id})

    def _parse_pair(self, pair_key: str) -> tuple[str, str] | None:
        members = [part.strip() for part in str(pair_key).split("|") if part.strip()]
        if len(members) != 2:
            return None
        left, right = sorted(members)
        return left, right

    def record_pairwise_agreements(self, pairs: Mapping[str, Any]) -> None:
        for pair_key, stats in pairs.items():
            if not isinstance(stats, Mapping):
                continue
            parsed = self._parse_pair(pair_key)
            if parsed is None:
                continue
            member_a, member_b = parsed
            for category in ("questions_together", "top_agreements", "agreement_rate"):
                value = stats.get(category)
                if isinstance(value, (int, float)):
                    self._set_gauge(
                        self._pairwise_agreement,
                        float(value),
                        labels={
                            "member_a": member_a,
                            "member_b": member_b,
                            "category": category,
                        },
                    )

    def record_collusion_warnings(self, warnings: Sequence[str]) -> None:
        for warning in warnings:
            if not isinstance(warning, str):
                continue
            match = re.search(r"between ([^\s]+)", warning)
            pair_key = match.group(1) if match else warning
            parsed = self._parse_pair(pair_key)
            if parsed is None:
                continue
            member_a, member_b = parsed
            self._increment(
                self._collusion_warnings_total,
                labels={"member_a": member_a, "member_b": member_b},
                amount=1,
            )


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
    if isinstance(metadata, Mapping):
        for key in ("agent_id", "originator_id", "requester_id", "author_id"):
            value = metadata.get(key)
            if value:
                return str(value)

    if question.user_id:
        return str(question.user_id)

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
    allow_remote_models = bool(raw_config.get("allow_remote_models", False))
    allowed_prefixes = _get_allowed_local_model_prefixes()
    default_model = _resolve_default_model(
        allow_remote=allow_remote_models, allowed_prefixes=allowed_prefixes
    )
    judge_model = str(raw_config.get("judge_model") or default_model)
    judge_model = _guard_local_model(
        judge_model,
        allow_remote=allow_remote_models,
        allowed_prefixes=allowed_prefixes,
        context="council judge",
    )
    voting_mode = str(raw_config.get("voting_mode") or "judge_llm")
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
        member_id = str(entry.get("member_id") or entry.get("id") or _slugify(display_name))
        role = str(entry.get("role") or "Generalist")
        persona = str(
            entry.get("persona")
            or entry.get("description")
            or f"{display_name} focuses on the {role} perspective."
        )
        system_prompt = str(
            entry.get("system_prompt")
            or (
                f"You are {display_name} ({role}). Embrace this persona: {persona}. "
                "Stay in-character while keeping answers concise and grounded."
            )
        )
        metadata = dict(entry.get("metadata") or {})
        model_name = str(entry.get("model") or metadata.get("model") or default_model)
        model_name = _guard_local_model(
            model_name,
            allow_remote=allow_remote_models,
            allowed_prefixes=allowed_prefixes,
            context=f"council member {member_id}",
        )
        metadata.setdefault("model", model_name)
        max_tokens = entry.get("max_tokens") or entry.get("max_turn_tokens")

        member_payload = {
            **entry,
            "member_id": member_id,
            "display_name": display_name,
            "role": role,
            "persona": persona,
            "system_prompt": system_prompt,
            "model": model_name,
            "temperature": float(entry.get("temperature", 0.3)),
            "max_tokens": int(max_tokens) if max_tokens is not None else None,
            "metadata": metadata,
        }

        members.append(CouncilMemberConfig.model_validate(member_payload))

    council_config = CouncilConfig.model_validate(
        {
            "enabled": bool(raw_config.get("enabled", True)),
            "voting_mode": voting_mode,
            "members": members,
            "quorum": raw_config.get("quorum"),
            "consensus_threshold": float(raw_config.get("consensus_threshold", 0.67)),
            "max_rounds": int(raw_config.get("max_rounds", 1)),
            "auto_record_transcript": bool(raw_config.get("auto_record_transcript", True)),
            "max_concurrent_calls": int(max_concurrent_calls) if max_concurrent_calls else None,
            "du_budget_per_question": (
                float(du_budget_per_question) if du_budget_per_question is not None else None
            ),
            "metadata": {"raw_config": raw_config},
            "allow_remote_models": allow_remote_models,
        }
    )

    return CouncilContext(
        config=council_config,
        judge_model=judge_model,
        member_model=default_model,
        allow_remote_models=allow_remote_models,
    )


def _get_allowed_local_model_prefixes() -> list[str]:
    prefixes: list[str] = []
    for key in ("LLM_API_BASE", "OLLAMA_API_BASE", "VLLM_API_BASE"):
        value = str(get_config(key) or "").strip()
        if value:
            prefixes.append(value.rstrip("/"))
    prefixes.extend(
        [
            "http://localhost",
            "https://localhost",
            "http://127.0.0.1",
            "https://127.0.0.1",
            "http://0.0.0.0",
            "https://0.0.0.0",
        ]
    )
    return prefixes


def _is_remote_model(model_name: str, allowed_prefixes: Sequence[str]) -> bool:
    normalized = model_name.strip()
    for prefix in allowed_prefixes:
        if normalized.startswith(prefix):
            return False

    lowered = normalized.lower()
    if lowered.startswith(("http://", "https://")):
        return True
    if "/" in normalized:
        return True
    return False


def _guard_local_model(
    model_name: str,
    *,
    allow_remote: bool,
    allowed_prefixes: Sequence[str] | None = None,
    context: str = "council",
) -> str:
    normalized = str(model_name).strip()
    if not normalized:
        raise ValueError(f"{context} model must be configured.")

    prefixes = list(allowed_prefixes or _get_allowed_local_model_prefixes())
    if _is_remote_model(normalized, prefixes):
        message = (
            f"Remote model '{normalized}' detected for {context}; "
            "configure a local model or enable allow_remote_models."
        )
        if allow_remote:
            logger.warning(message)
        else:
            raise ValueError(message)
    return normalized


def _resolve_default_model(
    *, allow_remote: bool = False, allowed_prefixes: Sequence[str] | None = None
) -> str:
    default_model = get_config("DEFAULT_LLM_MODEL")
    if not default_model or str(default_model).strip() == "":
        raise RuntimeError("DEFAULT_LLM_MODEL must be configured for council orchestration.")
    return _guard_local_model(
        str(default_model),
        allow_remote=allow_remote,
        allowed_prefixes=allowed_prefixes,
        context="council default",
    )


def _format_rag_docs(rag_docs: Sequence[str]) -> str:
    if not rag_docs:
        return "- (no retrieved documents; placeholder RAG list)"
    return "\n".join(f"- {doc}" for doc in rag_docs)


def _stringify_extra_context(extra_context: Mapping[str, Any] | None) -> str:
    if not extra_context:
        return "(no extra context provided)"
    text = extra_context.get("text")
    if isinstance(text, str) and text:
        return text
    summary = extra_context.get("summary")
    if isinstance(summary, str) and summary:
        return summary
    try:
        return json.dumps(extra_context, ensure_ascii=False)
    except TypeError:
        return str(extra_context)


def _build_member_prompt(
    member: CouncilMemberConfig,
    question: CouncilQuestion,
    *,
    extra_context: Mapping[str, Any] | None = None,
    rag_docs: Sequence[str] | None = None,
) -> str:
    rag_section = _format_rag_docs(rag_docs or [])
    additional_context = _stringify_extra_context(extra_context)
    return (
        "[council-member-answer] "
        f"member_id={member.member_id} question={question.prompt} context={question.context or ''} "
        f"extra_context={additional_context} rag_docs={rag_section}\n"
        f"System prompt for {member.display_name} ({member.role}): {member.system_prompt}\n"
        f"Persona description: {member.description}\n\n"
        f"{DEFAULT_MEMBER_PROMPT}"
    )


def _resolve_member_generation_params(member: CouncilMemberConfig) -> dict[str, float | int]:
    temperature = getattr(member, "temperature", None)
    if temperature is None:
        temperature_value = 0.3
    else:
        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError):
            temperature_value = 0.3

    max_tokens = getattr(member, "max_tokens", None)
    if max_tokens is None:
        max_tokens_value = 256
    else:
        try:
            max_tokens_value = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens_value = 256

    return {"temperature": temperature_value, "max_tokens": max_tokens_value}


def _ask_council_member(
    member: CouncilMemberConfig,
    question: CouncilQuestion,
    *,
    extra_context: Mapping[str, Any] | None = None,
    rag_docs: Sequence[str] | None = None,
    agent_state: Any | None = None,
) -> MemberAnswer:
    prompt = _build_member_prompt(member, question, extra_context=extra_context, rag_docs=rag_docs)
    model_name = (member.metadata or {}).get("model") if member.metadata else None
    member_model = str(model_name).strip() if model_name else ""
    if not member_model:
        member_model = _resolve_default_model()
    track_mock_usage = is_mock_mode_enabled()
    errors: list[str] = []
    generation_params = _resolve_member_generation_params(member)

    structured: MemberResponseModel | None = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            if track_mock_usage:
                llm_client.client.generate(prompt=prompt)
                structured = generate_structured_output(
                    prompt,
                    response_model=MemberResponseModel,
                    model=member_model,
                    temperature=generation_params["temperature"],
                    max_tokens=generation_params["max_tokens"],
                    agent_state=agent_state,
                )
            else:
                structured = generate_structured_output(
                    prompt,
                    response_model=MemberResponseModel,
                    model=member_model,
                    temperature=generation_params["temperature"],
                    max_tokens=generation_params["max_tokens"],
                    agent_state=agent_state,
                )
            if structured is not None:
                break
        except Exception as exc:  # pragma: no cover - defensive
            if isinstance(exc, RuntimeError) and "budget" in str(exc).lower():
                raise
            errors.append(str(exc))
            logger.warning(
                "Council member structured response failed",
                extra={
                    "event": "council.member.llm_error",
                    "member_id": member.member_id,
                    "attempt": attempt,
                },
                exc_info=True,
            )

    fallback_text = ""
    if structured is None:
        try:
            fallback_text = (
                generate_text(
                    prompt,
                    model=member_model,
                    temperature=generation_params["temperature"],
                    max_tokens=generation_params["max_tokens"],
                    agent_state=agent_state,
                )
                or ""
            )
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc))
            logger.error(
                "Council member fallback text generation failed",
                extra={
                    "event": "council.member.fallback_error",
                    "member_id": member.member_id,
                },
                exc_info=True,
            )
        return MemberAnswer(
            member_id=member.member_id,
            answer=fallback_text,
            reasoning=None,
            confidence=None,
            citations=[],
            metadata={"errors": errors, "fallback_used": True},
        )

    return MemberAnswer(
        member_id=member.member_id,
        answer=structured.answer,
        reasoning=structured.reasoning,
        confidence=structured.confidence,
        citations=structured.citations,
        metadata={"errors": errors, "fallback_used": False},
    )


def _build_judge_prompt(
    question: CouncilQuestion,
    answers: Sequence[MemberAnswer],
    *,
    extra_context: Mapping[str, Any] | None = None,
    rag_docs: Sequence[str] | None = None,
) -> str:
    rag_section = _format_rag_docs(rag_docs or [])
    additional_context = _stringify_extra_context(extra_context)
    member_blocks = []
    for answer in answers:
        member_blocks.append(
            "\n".join(
                [
                    f"- id: {answer.member_id}",
                    f"  answer: {answer.answer}",
                    f"  reasoning: {answer.reasoning or '(no reasoning provided)'}",
                ]
            )
        )
    member_section = "\n".join(member_blocks) or "(no member answers)"
    score_categories = ", ".join(JUDGE_SCORE_CATEGORIES)
    return "\n".join(
        [
            "[council-judgement]",
            f"Question: {question.prompt}",
            f"Context: {question.context or ''}",
            f"Extra context: {additional_context}",
            f"RAG docs: {rag_section}",
            "Members:",
            member_section,
            "",
            "Scoring guidance:",
            f"- Score each member from 0.0 to 1.0 for: {score_categories}.",
            "- Compute a total score as the average of the category scores.",
            "- Choose the winner with the highest total score (break ties by clarity).",
            "",
            "Return ONLY valid JSON matching this schema:",
            "{",
            '  "winner_id": "<member_id>",',
            '  "votes": {"<member_id>": 0.0},',
            '  "scores": {"<member_id>": {"correctness": 0.0, "clarity": 0.0, "usefulness": 0.0, "safety": 0.0}},',
            '  "metrics": {"cohesion": 0.0, "coverage": 0.0, "member_scores": {"<member_id>": {"correctness": 0.0, "clarity": 0.0, "usefulness": 0.0, "safety": 0.0, "total": 0.0}}},',
            '  "summary": "<concise summary>",',
            '  "reasoning": "<why the winner was selected>",',
            '  "resolution": "<final resolution or selected answer>"',
            "}",
        ]
    )


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_category_scores(raw: Mapping[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for category in JUDGE_SCORE_CATEGORIES:
        value = _coerce_score(raw.get(category))
        if value is not None:
            scores[category] = value
    return scores


def _normalize_vote_metrics(
    structured: CouncilVoteModel, answers: Sequence[MemberAnswer]
) -> CouncilVoteModel:
    totals: dict[str, float] = {}
    breakdown: dict[str, dict[str, float]] = {}

    for member_id, score_map in structured.scores.items():
        if isinstance(score_map, Mapping):
            category_scores = _extract_category_scores(score_map)
            if category_scores:
                breakdown[member_id] = category_scores

    for member_id, value in structured.votes.items():
        if isinstance(value, Mapping):
            category_scores = _extract_category_scores(value)
            if category_scores:
                breakdown.setdefault(member_id, category_scores)
        else:
            score_value = _coerce_score(value)
            if score_value is not None:
                totals[member_id] = score_value

    for member_id, category_scores in breakdown.items():
        if category_scores:
            totals.setdefault(member_id, sum(category_scores.values()) / len(category_scores))

    if not totals and breakdown:
        totals = {
            member_id: sum(scores.values()) / len(scores) for member_id, scores in breakdown.items()
        }

    member_ids = {answer.member_id for answer in answers}
    for member_id, total in totals.items():
        if member_id in member_ids:
            breakdown.setdefault(member_id, {})
            breakdown[member_id]["total"] = total

    structured.votes = totals
    if breakdown:
        structured.scores = {
            member_id: {
                category: score
                for category, score in scores.items()
                if category in JUDGE_SCORE_CATEGORIES
            }
            for member_id, scores in breakdown.items()
        }
        structured.metrics = {
            **structured.metrics,
            "member_scores": breakdown,
            "score_categories": list(JUDGE_SCORE_CATEGORIES),
        }
    return structured


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
    errors: list[str] = []
    structured: CouncilVoteModel | None = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            if track_mock_usage:
                llm_client.client.generate(prompt=prompt)
                structured = generate_structured_output(
                    prompt,
                    response_model=CouncilVoteModel,
                    model=context.judge_model,
                    temperature=0.1,
                )
            else:
                structured = generate_structured_output(
                    prompt,
                    response_model=CouncilVoteModel,
                    model=context.judge_model,
                    temperature=0.1,
                )
            if structured is not None:
                break
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc))
            logger.warning(
                "Council judge evaluation failed",
                extra={
                    "event": "council.judge.llm_error",
                    "question_id": question.question_id,
                    "attempt": attempt,
                },
                exc_info=True,
            )

    if structured is None and errors:
        logger.error(
            "Council judge failed after retries",
            extra={
                "event": "council.judge.failure",
                "question_id": question.question_id,
                "errors": errors,
            },
        )

    if structured is not None:
        structured = _normalize_vote_metrics(structured, answers)

    return structured


def _build_peer_vote_prompt(
    question: CouncilQuestion,
    answers: Sequence[MemberAnswer],
    *,
    voter: CouncilMemberConfig,
    extra_context: Mapping[str, Any] | None = None,
    rag_docs: Sequence[str] | None = None,
) -> str:
    rag_section = _format_rag_docs(rag_docs or [])
    additional_context = _stringify_extra_context(extra_context)
    member_blocks = []
    for answer in answers:
        member_blocks.append(
            "\n".join(
                [
                    f"- id: {answer.member_id}",
                    f"  answer: {answer.answer}",
                    f"  reasoning: {answer.reasoning or '(no reasoning provided)'}",
                    f"  confidence: {answer.confidence if answer.confidence is not None else 'n/a'}",
                ]
            )
        )
    member_section = "\n".join(member_blocks) or "(no member answers)"
    return "\n".join(
        [
            "[council-peer-vote]",
            f"Voter: {voter.member_id} ({voter.display_name})",
            f"Question: {question.prompt}",
            f"Context: {question.context or ''}",
            f"Extra context: {additional_context}",
            f"RAG docs: {rag_section}",
            "Members:",
            member_section,
            "",
            "Evaluate each member response. Provide a 0.0-1.0 score per member and select a winner.",
            "Return ONLY valid JSON matching this schema:",
            "{",
            '  "winner_id": "<member_id>",',
            '  "votes": {"<member_id>": 0.0},',
            '  "summary": "<concise summary>",',
            '  "reasoning": "<why the winner was selected>"',
            "}",
        ]
    )


def _ask_peer_vote(
    member: CouncilMemberConfig,
    question: CouncilQuestion,
    answers: Sequence[MemberAnswer],
    *,
    extra_context: Mapping[str, Any] | None = None,
    rag_docs: Sequence[str] | None = None,
    agent_state: Any | None = None,
) -> CouncilPeerVoteModel | None:
    prompt = _build_peer_vote_prompt(
        question, answers, voter=member, extra_context=extra_context, rag_docs=rag_docs
    )
    model_name = (member.metadata or {}).get("model") if member.metadata else None
    member_model = str(model_name).strip() if model_name else ""
    if not member_model:
        member_model = _resolve_default_model()
    track_mock_usage = is_mock_mode_enabled()
    errors: list[str] = []
    generation_params = _resolve_member_generation_params(member)

    structured: CouncilPeerVoteModel | None = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            if track_mock_usage:
                llm_client.client.generate(prompt=prompt)
                structured = generate_structured_output(
                    prompt,
                    response_model=CouncilPeerVoteModel,
                    model=member_model,
                    temperature=generation_params["temperature"],
                    max_tokens=generation_params["max_tokens"],
                    agent_state=agent_state,
                )
            else:
                structured = generate_structured_output(
                    prompt,
                    response_model=CouncilPeerVoteModel,
                    model=member_model,
                    temperature=generation_params["temperature"],
                    max_tokens=generation_params["max_tokens"],
                    agent_state=agent_state,
                )
            if structured is not None:
                break
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(str(exc))
            logger.warning(
                "Council peer vote failed",
                extra={
                    "event": "council.peer_vote.llm_error",
                    "member_id": member.member_id,
                    "attempt": attempt,
                },
                exc_info=True,
            )

    if structured is None and errors:
        logger.error(
            "Council peer vote failed after retries",
            extra={
                "event": "council.peer_vote.failure",
                "member_id": member.member_id,
                "errors": errors,
            },
        )
    return structured


def _select_winner_id(
    answers: Sequence[MemberAnswer],
    scores: Mapping[str, float],
) -> str:
    if not scores:
        return ""
    answer_lookup = {answer.member_id: answer for answer in answers}
    order_lookup = {answer.member_id: index for index, answer in enumerate(answers)}

    def _tie_key(member_id: str, score: float) -> tuple[float, float, int, int]:
        answer = answer_lookup.get(member_id)
        confidence = answer.confidence if answer and answer.confidence is not None else 0.5
        length = len(answer.answer) if answer else 0
        order = -order_lookup.get(member_id, 0)
        return (score, confidence, length, order)

    return max(scores.items(), key=lambda item: _tie_key(item[0], item[1]))[0]


def _build_vote_from_scores(
    answers: Sequence[MemberAnswer],
    *,
    votes: dict[str, float],
    summary: str,
    reasoning: str | None,
    metrics: dict[str, Any],
) -> CouncilVoteModel | None:
    winner_id = _select_winner_id(answers, votes)
    if not winner_id:
        return None
    answer_lookup = {answer.member_id: answer.answer for answer in answers}
    resolution = answer_lookup.get(winner_id)
    return CouncilVoteModel(
        winning_member_id=winner_id,
        votes=votes,
        scores={},
        metrics=metrics,
        summary=summary,
        reasoning=reasoning,
        resolution=resolution,
    )


def _aggregate_peer_votes(
    answers: Sequence[MemberAnswer],
    peer_votes: Sequence[tuple[CouncilMemberConfig, CouncilPeerVoteModel]],
) -> CouncilVoteModel | None:
    if not peer_votes:
        return None
    valid_member_ids = {answer.member_id for answer in answers}
    totals: dict[str, float] = {}
    per_voter: dict[str, dict[str, float]] = {}
    weights: dict[str, float] = {}

    for voter, vote in peer_votes:
        weight = float(voter.decision_weight or 1.0)
        weights[voter.member_id] = weight
        vote_scores: dict[str, float] = {}
        for member_id, raw_score in vote.votes.items():
            score = _coerce_score(raw_score)
            if score is None:
                continue
            member_id_str = str(member_id)
            if member_id_str not in valid_member_ids:
                continue
            vote_scores[member_id_str] = score
        if not vote_scores and vote.winner_id:
            winner_id = str(vote.winner_id)
            if winner_id in valid_member_ids:
                vote_scores[winner_id] = 1.0
        if not vote_scores:
            continue
        per_voter[voter.member_id] = vote_scores
        for member_id, score in vote_scores.items():
            totals[member_id] = totals.get(member_id, 0.0) + score * weight

    if not totals:
        return None

    metrics = {
        "peer_vote": {
            "voter_count": len(per_voter),
            "votes": per_voter,
            "weights": weights,
        }
    }
    summary = "Peer vote aggregation complete."
    reasoning = "Aggregated peer votes selected the highest-scoring response."
    return _build_vote_from_scores(
        answers,
        votes=totals,
        summary=summary,
        reasoning=reasoning,
        metrics=metrics,
    )


def _score_heuristic_votes(
    answers: Sequence[MemberAnswer],
    member_lookup: Mapping[str, CouncilMemberConfig],
) -> CouncilVoteModel | None:
    if not answers:
        return None
    totals: dict[str, float] = {}
    for answer in answers:
        confidence = answer.confidence if answer.confidence is not None else 0.5
        member = member_lookup.get(answer.member_id)
        weight = float(member.decision_weight) if member else 1.0
        totals[answer.member_id] = confidence * weight

    metrics = {
        "heuristic": {
            "scores": dict(totals),
            "tie_breaker": "confidence, answer length, member order",
        }
    }
    summary = "Heuristic scoring applied to council answers."
    reasoning = "Confidence-weighted scores with deterministic tie-breakers."
    return _build_vote_from_scores(
        answers,
        votes=totals,
        summary=summary,
        reasoning=reasoning,
        metrics=metrics,
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
            allow_remote_models=config.allow_remote_models,
        )

    def deliberate(
        self,
        config: CouncilConfig | None,
        question: CouncilQuestion,
        *,
        extra_context: Mapping[str, Any] | None = None,
        rag_docs: Sequence[str] | None = None,
        allow_disabled_mode: bool = False,
    ) -> CouncilOutcome:
        """Synchronously deliberate by awaiting the async implementation."""

        context = self._resolve_context(config)
        return asyncio.run(
            self.adeliberate(
                context,
                question,
                extra_context=extra_context,
                rag_docs=rag_docs,
                allow_disabled_mode=allow_disabled_mode,
            )
        )

    async def adeliberate(
        self,
        context: CouncilContext,
        question: CouncilQuestion,
        *,
        extra_context: Mapping[str, Any] | None = None,
        rag_docs: Sequence[str] | None = None,
        allow_disabled_mode: bool = False,
    ) -> CouncilOutcome:
        council_enabled = bool(get_config("USE_COUNCIL_MODE"))
        if not allow_disabled_mode and not council_enabled:
            raise RuntimeError(
                "Council mode is disabled; set USE_COUNCIL_MODE=true to enable it."
            )

        if not context.config.enabled:
            raise RuntimeError("Council mode is disabled in the council configuration.")

        if not context.config.members:
            raise ValueError("Council configuration must include at least one member")

        run_metrics = CouncilRunMetrics()
        run_metrics.record_run()
        total_start = time.perf_counter()

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
        if not context.config.members:
            metrics.update({"member_count": 0, "error": "no_active_members"})
            logger.warning(
                "No active council members available",
                extra={
                    "event": "council.no_members",
                    "question_id": question.question_id,
                },
            )
            run_metrics.record_latency("total", (time.perf_counter() - total_start) * 1000)
            return CouncilOutcome(
                question=question,
                answers=[],
                resolution="No active council members available",
                winning_member_ids=[],
                summary=None,
                metadata={"metrics": metrics},
            )
        member_states = self._allocate_du_budgets(context.config, metrics)

        member_fanout_start = time.perf_counter()
        answers = await self._gather_member_answers(
            context,
            question,
            member_states,
            extra_context=extra_context,
            rag_docs=rag_docs,
            metrics=metrics,
        )
        run_metrics.record_latency(
            "member_fanout", (time.perf_counter() - member_fanout_start) * 1000
        )

        if metrics.get("du_budget_exhausted"):
            metrics.setdefault("partial", True)
            metrics.setdefault("completed_members", [answer.member_id for answer in answers])
            run_metrics.record_du_usage(member_states, metrics.get("du_budget_per_member"))
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
                    "metrics": dict(metrics),
                },
            )
            run_metrics.record_latency("total", (time.perf_counter() - total_start) * 1000)
            await self._record_outcome_metrics(outcome)
            return outcome

        vote = None
        if answers:
            judge_start = time.perf_counter()
            if context.config.voting_mode == "judge_llm":
                vote = await asyncio.to_thread(
                    _judge_council_answers,
                    context,
                    question,
                    answers,
                    extra_context=extra_context,
                    rag_docs=rag_docs,
                )
            elif context.config.voting_mode == "peer_vote":
                peer_votes = await self._gather_peer_votes(
                    context,
                    question,
                    answers,
                    member_states,
                    extra_context=extra_context,
                    rag_docs=rag_docs or [],
                    metrics=metrics,
                )
                vote = _aggregate_peer_votes(answers, peer_votes)
            elif context.config.voting_mode == "heuristic":
                member_lookup = {member.member_id: member for member in context.config.members}
                vote = _score_heuristic_votes(answers, member_lookup)
            else:
                logger.warning(
                    "Unknown council voting mode '%s'; skipping vote.",
                    context.config.voting_mode,
                )
            run_metrics.record_latency("judge", (time.perf_counter() - judge_start) * 1000)

        run_metrics.record_member_scores_from_vote(vote)
        run_metrics.record_du_usage(member_states, metrics.get("du_budget_per_member"))

        base_metrics = dict(metrics or {})
        base_metrics.setdefault("du_budget_exhausted", False)
        base_metrics.setdefault(
            "du_budget_per_member", self._resolve_du_budget(context.config)
        )
        outcome = self._build_outcome(
            question, answers, vote, base_metrics, run_metrics=run_metrics
        )
        run_metrics.record_latency("total", (time.perf_counter() - total_start) * 1000)
        await self._record_outcome_metrics(outcome)
        return outcome

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
        extra_context: Mapping[str, Any] | None,
        rag_docs: Sequence[str],
        metrics: dict[str, Any],
    ) -> list[MemberAnswer]:
        max_concurrent = max(
            1, context.config.max_concurrent_calls or self.max_concurrent_calls
        )
        semaphore = asyncio.Semaphore(max_concurrent)
        failures: list[str] = []
        answers: list[MemberAnswer] = []

        def _has_available_budget(
            member: CouncilMemberConfig, state: SimpleNamespace | None
        ) -> bool:
            try:
                resource_manager = get_resource_manager()
            except Exception:
                resource_manager = None

            if resource_manager is not None:
                try:
                    if resource_manager.get_du_budget(member.member_id) <= 0:
                        return False
                except Exception:
                    pass

            if state is not None:
                remaining = getattr(state, "du", None)
                if remaining is not None and remaining <= 0:
                    return False

            return True

        async def _call_member(
            member: CouncilMemberConfig,
        ) -> tuple[str, MemberAnswer | None]:
            state = member_states.get(member.member_id)
            try:
                async with semaphore:
                    result = await asyncio.to_thread(
                        _ask_council_member,
                        member,
                        question,
                        extra_context=extra_context,
                        rag_docs=rag_docs,
                        agent_state=state,
                    )
                    return member.member_id, result
            except RuntimeError as exc:
                if "budget" in str(exc).lower():
                    metrics["du_budget_exhausted"] = True
                    logger.warning(
                        "DU budget exhausted for member %s: %s", member.member_id, exc
                    )
                else:
                    logger.exception("Council member call failed: %s", exc)
                return member.member_id, None
            except Exception as exc:
                if "budget" in str(exc).lower():
                    metrics["du_budget_exhausted"] = True
                else:
                    logger.exception("Unexpected error during council call", exc_info=exc)
                return member.member_id, None

        async def _drain_tasks(
            tasks: list[asyncio.Task[tuple[str, MemberAnswer | None]]],
        ) -> None:
            if not tasks:
                return
            results = await asyncio.gather(*tasks)
            tasks.clear()
            for member_id, result in results:
                if result is None:
                    failures.append(member_id)
                else:
                    answers.append(result)

        tasks: list[asyncio.Task[tuple[str, MemberAnswer | None]]] = []

        for member in context.config.members:
            if metrics.get("du_budget_exhausted"):
                break
            state = member_states.get(member.member_id)
            if not _has_available_budget(member, state):
                metrics["du_budget_exhausted"] = True
                logger.warning(
                    "DU budget exhausted before scheduling member %s", member.member_id
                )
                break
            tasks.append(asyncio.create_task(_call_member(member)))
            if len(tasks) >= max_concurrent:
                await _drain_tasks(tasks)
                if metrics.get("du_budget_exhausted"):
                    break

        if tasks:
            await _drain_tasks(tasks)

        if failures:
            metrics["partial"] = True
            metrics["failed_members"] = sorted(set(failures))

        if metrics.get("du_budget_exhausted"):
            if answers:
                metrics.setdefault(
                    "completed_members", [answer.member_id for answer in answers]
                )
            metrics.setdefault("partial", True)
        return answers

    async def _gather_peer_votes(
        self,
        context: CouncilContext,
        question: CouncilQuestion,
        answers: Sequence[MemberAnswer],
        member_states: Mapping[str, SimpleNamespace],
        *,
        extra_context: Mapping[str, Any] | None,
        rag_docs: Sequence[str],
        metrics: dict[str, Any],
    ) -> list[tuple[CouncilMemberConfig, CouncilPeerVoteModel]]:
        semaphore = asyncio.Semaphore(
            max(1, context.config.max_concurrent_calls or self.max_concurrent_calls)
        )
        failures: list[str] = []

        async def _call_member(member: CouncilMemberConfig) -> tuple[
            CouncilMemberConfig, CouncilPeerVoteModel
        ] | None:
            state = member_states.get(member.member_id)
            try:
                async with semaphore:
                    result = await asyncio.to_thread(
                        _ask_peer_vote,
                        member,
                        question,
                        answers,
                        extra_context=extra_context,
                        rag_docs=rag_docs,
                        agent_state=state,
                    )
                    if result is None:
                        return None
                    return (member, result)
            except RuntimeError as exc:
                if "budget" in str(exc).lower():
                    metrics["du_budget_exhausted"] = True
                    logger.warning(
                        "DU budget exhausted during peer vote for %s: %s",
                        member.member_id,
                        exc,
                    )
                else:
                    logger.exception("Council peer vote failed: %s", exc)
                    failures.append(member.member_id)
                return None

        results = await asyncio.gather(
            *[_call_member(member) for member in context.config.members],
            return_exceptions=True,
        )

        votes: list[tuple[CouncilMemberConfig, CouncilPeerVoteModel]] = []
        for member, result in zip(context.config.members, results):
            if isinstance(result, Exception):
                if "budget" in str(result).lower():
                    metrics["du_budget_exhausted"] = True
                else:
                    logger.exception("Unexpected peer vote error", exc_info=result)
                    failures.append(member.member_id)
                continue
            if result is None:
                failures.append(member.member_id)
                continue
            votes.append(result)

        if failures:
            metrics["peer_vote_partial"] = True
            metrics["peer_vote_failed_members"] = sorted(set(failures))
        return votes

    def _build_outcome(
        self,
        question: CouncilQuestion,
        answers: list[MemberAnswer],
        vote: CouncilVoteModel | None,
        metrics: Mapping[str, Any],
        run_metrics: CouncilRunMetrics | None = None,
    ) -> CouncilOutcome:
        winning_member_ids: list[str] = []
        resolution = "No consensus reached."
        summary = None
        outcome_metrics: dict[str, Any] = dict(metrics)
        votes: dict[str, float] = {}
        metadata: dict[str, Any] = {}
        fitness_snapshot: Mapping[str, Any] | None = None
        winner_answer: str | None = None

        if vote is not None:
            winning_member_ids = [vote.winning_member_id]
            votes = dict(vote.votes)
            outcome_metrics.update(vote.metrics)
            metadata.update({"scores": vote.scores or votes, "judge_reasoning": vote.reasoning})
            summary = vote.summary
            answer_lookup = {answer.member_id: answer.answer for answer in answers}
            resolution = vote.resolution or answer_lookup.get(
                vote.winning_member_id, resolution
            )
            winner_answer = answer_lookup.get(vote.winning_member_id)
            fitness_start = time.perf_counter()
            fitness_snapshot = self.fitness_store.update_from_vote(
                question, answers, vote
            )
            if run_metrics is not None:
                run_metrics.record_latency(
                    "fitness_update", (time.perf_counter() - fitness_start) * 1000
                )
                run_metrics.record_fitness_snapshot(fitness_snapshot)

        if fitness_snapshot is not None:
            metadata["fitness"] = fitness_snapshot
            outcome_metrics["fitness_snapshot"] = fitness_snapshot

        metadata["metrics"] = outcome_metrics

        return CouncilOutcome(
            question=question,
            answers=answers,
            resolution=resolution,
            winner_id=winning_member_ids[0] if winning_member_ids else None,
            winning_member_ids=winning_member_ids,
            winner_answer=winner_answer,
            votes=votes,
            metrics=outcome_metrics,
            summary=summary,
            metadata=metadata,
        )

    async def _record_outcome_metrics(self, outcome: CouncilOutcome) -> None:
        try:
            await council_stats_store.record_outcome_async(outcome)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to persist council metrics",
                extra={
                    "event": "council.metrics.error",
                    "question_id": outcome.question.question_id,
                },
                exc_info=True,
            )

    def serialize_metrics(self, *, question_id: str | None = None) -> dict[str, list[dict[str, float]]]:
        """Return a snapshot of aggregated council metrics."""

        return council_stats_store.serialize_metrics(question_id=question_id)

    async def serialize_metrics_async(
        self, *, question_id: str | None = None
    ) -> dict[str, list[dict[str, float]]]:
        return await council_stats_store.serialize_metrics_async(question_id=question_id)


def run_council(
    question: CouncilQuestion,
    *,
    extra_context: Mapping[str, Any] | None = None,
    rag_docs: Sequence[str] | None = None,
    allow_disabled_mode: bool = False,
) -> CouncilOutcome:
    """Gather answers from council members and select a winner using a judge model."""

    orchestrator = CouncilOrchestrator()
    return orchestrator.deliberate(
        _build_council_context().config,
        question,
        extra_context=extra_context,
        rag_docs=rag_docs,
        allow_disabled_mode=allow_disabled_mode,
    )
