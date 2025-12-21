from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

import src.agents.council.orchestrator as council_orchestrator
from src.agents.council import (
    CouncilConfig,
    CouncilMemberConfig,
    CouncilOrchestrator,
    CouncilQuestion,
)
from src.agents.council.fitness_store import council_fitness_store
from src.agents.council.stats_store import CouncilStatsStore
from src.infra import config, llm_client
from src.shared import llm_mocks

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_mocks.patch_ollama_functions(monkeypatch)
    llm_client.enable_mock_mode(
        True,
        {
            "CouncilVoteModel": {
                "winner_id": "facilitator",
                "winning_member_id": "facilitator",
                "votes": {},
                "scores": {
                    "facilitator": {
                        "correctness": 0.9,
                        "clarity": 0.8,
                        "usefulness": 0.7,
                        "safety": 0.95,
                    },
                    "innovator": {
                        "correctness": 0.6,
                        "clarity": 0.7,
                        "usefulness": 0.8,
                        "safety": 0.9,
                    },
                    "analyst": {
                        "correctness": 0.7,
                        "clarity": 0.6,
                        "usefulness": 0.85,
                        "safety": 0.9,
                    },
                },
                "metrics": {"cohesion": 0.91, "coverage": 0.77},
                "summary": "Deterministic council summary",
                "reasoning": "deterministic reasoning",
                "resolution": "facilitator proposal selected",
            },
            "MemberResponseModel": {
                "answer": "facilitator proposal selected",
                "reasoning": "deterministic reasoning",
                "confidence": 0.82,
                "citations": ["citation-1"],
            },
        },
    )


@pytest.fixture(autouse=True)
def reset_fitness_store() -> None:
    council_fitness_store.reset()


@pytest.fixture(autouse=True)
def enable_council_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_CONFIG", {"USE_COUNCIL_MODE": True})


def _build_council_config(num_members: int = 3) -> CouncilConfig:
    base_members = [
        CouncilMemberConfig(
            member_id="facilitator",
            display_name="Facilitator",
            role="Facilitator",
            description="Ensures everyone is heard",
            system_prompt="Lead with clarity",
            decision_weight=1.0,
            persona="Guides the conversation",
            model="mistral:latest",
            temperature=0.2,
            max_tokens=256,
            is_active=True,
        ),
        CouncilMemberConfig(
            member_id="innovator",
            display_name="Innovator",
            role="Innovator",
            description="Pushes creative thinking",
            system_prompt="Bring new ideas",
            decision_weight=1.0,
            persona="Explores creative options",
            model="mistral:latest",
            temperature=0.3,
            max_tokens=256,
            is_active=True,
        ),
        CouncilMemberConfig(
            member_id="analyst",
            display_name="Analyst",
            role="Analyzer",
            description="Stress-tests ideas",
            system_prompt="Look for gaps",
            decision_weight=1.0,
            persona="Evaluates trade-offs",
            model="mistral:latest",
            temperature=0.25,
            max_tokens=256,
            is_active=True,
        ),
    ]

    extra_members: list[CouncilMemberConfig] = []
    for idx in range(3, num_members):
        extra_members.append(
            CouncilMemberConfig(
                member_id=f"member-{idx}",
                display_name=f"Member {idx}",
                role="Generalist",
                description="Brings domain expertise",
                system_prompt="Share focused insight",
                decision_weight=1.0,
                persona="Subject matter expert",
                model="mistral:latest",
                temperature=0.3,
                max_tokens=256,
                is_active=True,
            )
        )

    return CouncilConfig(enabled=True, members=base_members + extra_members)


def _build_question(metadata: Mapping[str, Any] | None = None) -> CouncilQuestion:
    return CouncilQuestion(
        question_id="q-1",
        prompt="Which project should we fund first?",
        context="We have budget for only one initiative this quarter.",
        metadata=metadata,
    )


class DummyRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int | None]] = []

    async def retrieve(
        self, agent_id: str, query: str = "", k: int = 5, token_budget: int | None = None
    ) -> list[dict[str, str]]:
        self.calls.append((agent_id, query, k, token_budget))
        return [
            {"content": "Memory fact", "metadata": {"source": "episodic-1"}},
            {"content": "Semantic insight"},
        ]


def test_council_orchestrator_invokes_all_members_and_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator()
    config = _build_council_config()
    question = _build_question()

    outcome = orchestrator.deliberate(config, question)

    expected_member_ids = {member.member_id for member in config.members}
    observed_member_ids = {answer.member_id for answer in outcome.answers}

    assert observed_member_ids == expected_member_ids

    assert outcome.winning_member_ids == ["facilitator"]
    assert outcome.winner_id == "facilitator"
    assert outcome.resolution == "facilitator proposal selected"
    assert outcome.votes["facilitator"] == pytest.approx(0.8375)
    assert outcome.metadata is not None
    metrics = outcome.metadata.get("metrics", {})
    assert metrics.get("du_budget_exhausted") is False
    assert metrics.get("du_budget_per_member") == pytest.approx(5.0)
    assert "member_scores" in metrics
    assert outcome.metrics["member_scores"]["facilitator"]["total"] == pytest.approx(0.8375)
    assert outcome.summary == "Deterministic council summary"

    serialized = json.loads(json.dumps(outcome.metadata))
    assert serialized["metrics"]["cohesion"] == pytest.approx(0.91)
    fitness = serialized.get("fitness", {})
    member_fitness = fitness.get("members", {})
    assert member_fitness["facilitator"]["wins"] == 1
    assert member_fitness["facilitator"]["win_rate"] == pytest.approx(1.0)
    assert not fitness.get("warnings")


def test_council_orchestrator_limits_concurrent_generate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator(max_concurrency=2)
    config = _build_council_config(num_members=5)
    question = _build_question()

    llm_mocks.mock_generate_stats.reset()

    outcome = orchestrator.deliberate(config, question)

    assert outcome.winning_member_ids
    assert llm_mocks.mock_generate_stats.peak_concurrent_calls <= orchestrator.max_concurrency
    assert llm_mocks.mock_generate_stats.call_count == len(config.members) + 1


def test_council_orchestrator_marks_du_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = CouncilOrchestrator(max_concurrency=3)
    config = _build_council_config(num_members=4)
    question = _build_question()

    llm_mocks.set_mock_llm_du_budget(2)

    outcome = orchestrator.deliberate(config, question)

    assert len(outcome.answers) == 2
    assert outcome.winning_member_ids == []
    assert outcome.metadata is not None
    assert outcome.metadata.get("du_exhausted") is True
    assert outcome.metadata.get("partial") is True
    assert outcome.metadata.get("completed_members") == ["facilitator", "innovator"]
    metrics = outcome.metadata.get("metrics", {})
    assert metrics.get("du_budget_exhausted") is True
    assert metrics.get("partial") is True
    assert metrics.get("completed_members") == ["facilitator", "innovator"]

    llm_mocks.set_mock_llm_du_budget(None)


def test_council_orchestrator_includes_rag_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = CouncilOrchestrator()
    config = _build_council_config()
    question = _build_question()

    rag_marker = "<mocked-rag-docs>"
    extra_context = "<extra-context>"
    member_prompts: list[str] = []
    judge_prompts: list[str] = []

    monkeypatch.setattr("src.agents.council.orchestrator._format_rag_docs", lambda _: rag_marker)

    original_generate = llm_client.client.generate

    def capture_generate(*args: object, **kwargs: object) -> dict[str, object]:
        prompt = str(kwargs.get("prompt") or (args[0] if args else ""))
        if "[council-member-answer]" in prompt:
            member_prompts.append(prompt)
        if "[council-judgement]" in prompt:
            judge_prompts.append(prompt)
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(llm_client.client, "generate", capture_generate)

    orchestrator.deliberate(
        config,
        question,
        extra_context=extra_context,
        rag_docs=["Doc 1", "Doc 2"],
    )

    assert member_prompts
    assert judge_prompts

    assert all(rag_marker in prompt for prompt in member_prompts)
    assert all(rag_marker in prompt for prompt in judge_prompts)
    assert all(extra_context in prompt for prompt in member_prompts)
    assert all(extra_context in prompt for prompt in judge_prompts)
    assert all("facilitator proposal selected" in prompt for prompt in judge_prompts)


def test_council_orchestrator_persists_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    store = CouncilStatsStore(db_path=tmp_path_factory.mktemp("council-metrics") / "stats.sqlite3")
    monkeypatch.setattr(council_orchestrator, "council_stats_store", store)

    orchestrator = CouncilOrchestrator()
    config = _build_council_config()
    question = _build_question()

    outcome = orchestrator.deliberate(config, question)

    snapshot = store.serialize_metrics()
    facilitator_stats = next(m for m in snapshot["members"] if m["member_id"] == "facilitator")
    assert facilitator_stats["wins"] == 1
    assert facilitator_stats["participations"] == 1
    assert len(snapshot["members"]) == len(config.members)

    pairwise_entries = {(p["member_a"], p["member_b"]): p for p in snapshot["pairwise"]}
    assert pairwise_entries[("analyst", "facilitator")]["agreements"] == 1


def test_council_orchestrator_flags_partial_metrics_on_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator()
    config = _build_council_config()
    question = _build_question()

    original = council_orchestrator._ask_council_member

    def _raise_on_innovator(member: CouncilMemberConfig, *args: object, **kwargs: object):
        if member.member_id == "innovator":
            raise RuntimeError("LLM failure for innovator")
        return original(member, *args, **kwargs)

    monkeypatch.setattr(council_orchestrator, "_ask_council_member", _raise_on_innovator)

    outcome = orchestrator.deliberate(config, question)

    metrics = outcome.metadata.get("metrics", {})
    assert metrics.get("partial") is True
    assert metrics.get("failed_members") == ["innovator"]
    assert len(outcome.answers) == len(config.members) - 1


def test_council_orchestrator_requires_enabled_council(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_CONFIG", {"USE_COUNCIL_MODE": False})
    orchestrator = CouncilOrchestrator()

    with pytest.raises(RuntimeError, match="Council mode is disabled"):
        orchestrator.deliberate(_build_council_config(), _build_question())


def test_council_orchestrator_requires_members(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = CouncilOrchestrator()

    with pytest.raises(ValidationError, match="at least one member"):
        CouncilConfig(enabled=True, members=[])
