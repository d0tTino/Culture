from __future__ import annotations

import json
import threading
import time
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
    MemberAnswer,
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
            "CouncilPeerVoteModel": {
                "winner_id": "facilitator",
                "votes": {"facilitator": 0.9, "innovator": 0.6, "analyst": 0.7},
                "summary": "Peer vote summary",
                "reasoning": "Peer vote reasoning",
            },
            "MemberResponseModel": {
                "answer": "facilitator proposal selected",
                "reasoning": "deterministic reasoning",
                "confidence": 0.82,
                "citations": ["citation-1"],
            },
        },
    )
    yield
    llm_client.enable_mock_mode(False)


@pytest.fixture(autouse=True)
def reset_fitness_store() -> None:
    council_fitness_store.reset()


@pytest.fixture(autouse=True)
def enable_council_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config,
        "_CONFIG",
        {"USE_COUNCIL_MODE": True, "DEFAULT_LLM_MODEL": "mistral:latest"},
    )
    monkeypatch.setattr(config, "_COUNCIL_CONFIG", None)
    monkeypatch.setattr(config.settings, "DEFAULT_LLM_MODEL", "mistral:latest")


def _build_council_config(num_members: int = 3, voting_mode: str = "judge_llm") -> CouncilConfig:
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

    return CouncilConfig(
        enabled=True, members=base_members + extra_members, voting_mode=voting_mode
    )


def _build_question(metadata: Mapping[str, Any] | None = None) -> CouncilQuestion:
    return CouncilQuestion(
        question_id="q-1",
        prompt="Which project should we fund first?",
        context="We have budget for only one initiative this quarter.",
        metadata=metadata,
    )


def test_council_member_forwards_generation_params(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_council_config()
    member = config.members[0]
    question = _build_question()
    captured: dict[str, float | int] = {}

    def fake_generate_structured_output(*args: object, **kwargs: object):
        captured["temperature"] = float(kwargs.get("temperature"))
        captured["max_tokens"] = int(kwargs.get("max_tokens"))
        return council_orchestrator.MemberResponseModel(
            answer="captured answer",
            reasoning="captured reasoning",
            confidence=0.5,
            citations=[],
        )

    monkeypatch.setattr(council_orchestrator, "generate_structured_output", fake_generate_structured_output)

    answer = council_orchestrator._ask_council_member(member, question)

    assert answer.answer == "captured answer"
    assert captured["temperature"] == pytest.approx(member.temperature)
    assert captured["max_tokens"] == member.max_tokens


def test_council_member_fallback_forwards_generation_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_council_config()
    member = config.members[1]
    question = _build_question()
    captured: dict[str, float | int] = {}

    def fake_generate_structured_output(*args: object, **kwargs: object):
        return None

    def fake_generate_text(*args: object, **kwargs: object) -> str:
        captured["temperature"] = float(kwargs.get("temperature"))
        captured["max_tokens"] = int(kwargs.get("max_tokens"))
        return "fallback answer"

    monkeypatch.setattr(council_orchestrator, "generate_structured_output", fake_generate_structured_output)
    monkeypatch.setattr(council_orchestrator, "generate_text", fake_generate_text)

    answer = council_orchestrator._ask_council_member(member, question)

    assert answer.answer == "fallback answer"
    assert captured["temperature"] == pytest.approx(member.temperature)
    assert captured["max_tokens"] == member.max_tokens


def test_peer_vote_forwards_generation_params(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_council_config()
    member = config.members[2]
    question = _build_question()
    answers = [
        MemberAnswer(member_id="facilitator", answer="Answer A"),
        MemberAnswer(member_id="innovator", answer="Answer B"),
    ]
    captured: dict[str, float | int] = {}

    def fake_generate_structured_output(*args: object, **kwargs: object):
        captured["temperature"] = float(kwargs.get("temperature"))
        captured["max_tokens"] = int(kwargs.get("max_tokens"))
        return council_orchestrator.CouncilPeerVoteModel(
            winner_id="facilitator",
            votes={"facilitator": 0.9},
            summary="Peer vote summary",
            reasoning="Peer vote reasoning",
        )

    monkeypatch.setattr(council_orchestrator, "generate_structured_output", fake_generate_structured_output)

    result = council_orchestrator._ask_peer_vote(member, question, answers)

    assert result is not None
    assert result.winner_id == "facilitator"
    assert captured["temperature"] == pytest.approx(member.temperature)
    assert captured["max_tokens"] == member.max_tokens


def test_council_orchestrator_can_bypass_env_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator()
    council_config = _build_council_config(voting_mode="judge_llm")
    question = _build_question()

    monkeypatch.setattr(
        config,
        "_CONFIG",
        {"USE_COUNCIL_MODE": False, "DEFAULT_LLM_MODEL": "http://localhost/mock"},
    )

    with pytest.raises(RuntimeError):
        orchestrator.deliberate(council_config, question)

    outcome = orchestrator.deliberate(
        council_config, question, allow_disabled_mode=True
    )

    assert outcome.winning_member_ids


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
    config = _build_council_config(voting_mode="judge_llm")
    question = _build_question()

    outcome = orchestrator.deliberate(config, question)

    expected_member_ids = {member.member_id for member in config.members}
    observed_member_ids = {answer.member_id for answer in outcome.answers}

    assert observed_member_ids == expected_member_ids

    assert outcome.winning_member_ids == ["facilitator"]
    assert outcome.winner_id == "facilitator"
    assert outcome.winner_answer == "facilitator proposal selected"
    assert outcome.resolution == "facilitator proposal selected"
    assert outcome.votes["facilitator"] == pytest.approx(0.8375)
    assert outcome.metadata is not None
    metrics = outcome.metadata.get("metrics", {})
    assert metrics.get("du_budget_exhausted") is False
    assert metrics.get("du_budget_per_member") == pytest.approx(5.0)
    assert "member_scores" in metrics
    assert outcome.metrics["member_scores"]["facilitator"]["total"] == pytest.approx(0.8375)
    assert outcome.summary == "Deterministic council summary"
    assert "fitness_snapshot" in outcome.metrics
    assert outcome.metrics["agreement_score"] == pytest.approx(1.0 / 3.0)
    assert outcome.metrics["collusion_warnings"] == []
    pairwise_ema = outcome.metrics["pairwise_ema"]
    assert set(pairwise_ema.keys()) == {
        "analyst|facilitator",
        "analyst|innovator",
        "facilitator|innovator",
    }
    pairwise_stats = outcome.metrics["pairwise_ema_stats"]
    pairwise_pairs = {(entry["member_a"], entry["member_b"]) for entry in pairwise_stats}
    assert pairwise_pairs == {
        ("analyst", "facilitator"),
        ("analyst", "innovator"),
        ("facilitator", "innovator"),
    }
    assert all(entry["ema_last_updated"] for entry in pairwise_stats)
    pairwise_by_pair = outcome.metrics["pairwise_ema_by_pair"]
    assert set(pairwise_by_pair.keys()) == {
        "analyst|facilitator",
        "analyst|innovator",
        "facilitator|innovator",
    }
    assert pairwise_by_pair["analyst|facilitator"]["ema_agreement"] >= 0.0

    serialized = json.loads(json.dumps(outcome.metadata))
    assert serialized["metrics"]["cohesion"] == pytest.approx(0.91)
    assert serialized["metrics"]["pairwise_ema_stats"] == pairwise_stats
    assert serialized["metrics"]["pairwise_ema_by_pair"] == pairwise_by_pair
    fitness = serialized.get("fitness", {})
    member_fitness = fitness.get("members", {})
    assert member_fitness["facilitator"]["wins"] == 1
    assert member_fitness["facilitator"]["win_rate"] == pytest.approx(1.0)
    assert not fitness.get("warnings")
    assert outcome.metrics["fitness_snapshot"]["members"]["facilitator"]["wins"] == 1


def test_council_orchestrator_limits_concurrent_generate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator(max_concurrency=2)
    config = _build_council_config(num_members=5, voting_mode="judge_llm")
    question = _build_question()

    llm_mocks.mock_generate_stats.reset()

    outcome = orchestrator.deliberate(config, question)

    assert outcome.winning_member_ids
    assert llm_mocks.mock_generate_stats.peak_concurrent_calls <= orchestrator.max_concurrency
    assert llm_mocks.mock_generate_stats.call_count == len(config.members) + 1


def test_council_orchestrator_runs_member_calls_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator(max_concurrency=3)
    config = _build_council_config(num_members=3, voting_mode="judge_llm")
    question = _build_question()

    barrier = threading.Barrier(len(config.members))
    delays = {"facilitator": 0.15, "innovator": 0.05, "analyst": 0.1}

    def _slow_member(member: CouncilMemberConfig, *args: object, **kwargs: object) -> MemberAnswer:
        barrier.wait(timeout=1.0)
        time.sleep(delays[member.member_id])
        return MemberAnswer(member_id=member.member_id, answer=f"Answer {member.member_id}")

    monkeypatch.setattr(council_orchestrator, "_ask_council_member", _slow_member)

    outcome = orchestrator.deliberate(config, question)

    assert [answer.member_id for answer in outcome.answers] == [
        member.member_id for member in config.members
    ]


def test_council_orchestrator_marks_du_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = CouncilOrchestrator(max_concurrency=3)
    config = _build_council_config(num_members=4, voting_mode="judge_llm")
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


def test_council_orchestrator_stops_calls_after_du_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator(max_concurrency=3)
    config = _build_council_config(num_members=4, voting_mode="judge_llm")
    question = _build_question()

    llm_mocks.mock_generate_stats.reset()
    llm_mocks.set_mock_llm_du_budget(2)

    outcome = orchestrator.deliberate(config, question)

    assert outcome.metadata is not None
    assert outcome.metadata.get("du_exhausted") is True
    assert llm_mocks.mock_generate_stats.call_count == 3

    llm_mocks.set_mock_llm_du_budget(None)


def test_council_orchestrator_peer_vote_du_exhaustion_stops_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator(max_concurrency=3)
    config = _build_council_config(num_members=4, voting_mode="peer_vote")
    question = _build_question()

    llm_mocks.mock_generate_stats.reset()
    llm_mocks.set_mock_llm_du_budget(2)

    member_calls: list[str] = []
    peer_vote_calls: list[str] = []

    original_member = council_orchestrator._ask_council_member
    original_peer = council_orchestrator._ask_peer_vote

    def _count_member(member: CouncilMemberConfig, *args: object, **kwargs: object):
        member_calls.append(member.member_id)
        return original_member(member, *args, **kwargs)

    def _count_peer_vote(member: CouncilMemberConfig, *args: object, **kwargs: object):
        peer_vote_calls.append(member.member_id)
        return original_peer(member, *args, **kwargs)

    monkeypatch.setattr(council_orchestrator, "_ask_council_member", _count_member)
    monkeypatch.setattr(council_orchestrator, "_ask_peer_vote", _count_peer_vote)

    outcome = orchestrator.deliberate(config, question)

    assert outcome.metadata is not None
    assert outcome.metadata.get("du_exhausted") is True
    assert outcome.winning_member_ids == []
    assert len(outcome.answers) == 2
    assert len(member_calls) == 3
    assert peer_vote_calls == []
    assert llm_mocks.mock_generate_stats.call_count == 3

    llm_mocks.set_mock_llm_du_budget(None)


def test_council_orchestrator_heuristic_du_exhaustion_stops_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = CouncilOrchestrator(max_concurrency=3)
    config = _build_council_config(num_members=4, voting_mode="heuristic")
    question = _build_question()

    llm_mocks.mock_generate_stats.reset()
    llm_mocks.set_mock_llm_du_budget(2)

    member_calls: list[str] = []
    original_member = council_orchestrator._ask_council_member

    def _count_member(member: CouncilMemberConfig, *args: object, **kwargs: object):
        member_calls.append(member.member_id)
        return original_member(member, *args, **kwargs)

    monkeypatch.setattr(council_orchestrator, "_ask_council_member", _count_member)

    outcome = orchestrator.deliberate(config, question)

    assert outcome.metadata is not None
    assert outcome.metadata.get("du_exhausted") is True
    assert outcome.winning_member_ids == []
    assert len(outcome.answers) == 2
    assert len(member_calls) == 3
    assert llm_mocks.mock_generate_stats.call_count == 3

    llm_mocks.set_mock_llm_du_budget(None)


def test_council_orchestrator_includes_rag_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = CouncilOrchestrator()
    config = _build_council_config(voting_mode="judge_llm")
    question = _build_question()

    rag_marker = "<mocked-rag-docs>"
    extra_context = {"text": "<extra-context>"}
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
    assert all("<extra-context>" in prompt for prompt in member_prompts)
    assert all("<extra-context>" in prompt for prompt in judge_prompts)
    assert all("facilitator proposal selected" in prompt for prompt in judge_prompts)


def test_council_orchestrator_persists_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    store = CouncilStatsStore(db_path=tmp_path_factory.mktemp("council-metrics") / "stats.sqlite3")
    monkeypatch.setattr(council_orchestrator, "council_stats_store", store)

    orchestrator = CouncilOrchestrator()
    config = _build_council_config(voting_mode="judge_llm")
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
    config = _build_council_config(voting_mode="judge_llm")
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
    monkeypatch.setattr(
        config,
        "_CONFIG",
        {"USE_COUNCIL_MODE": False, "DEFAULT_LLM_MODEL": "http://localhost/mock"},
    )
    orchestrator = CouncilOrchestrator()

    with pytest.raises(RuntimeError, match="Council mode is disabled"):
        orchestrator.deliberate(_build_council_config(), _build_question())


def test_council_orchestrator_requires_members(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = CouncilOrchestrator()

    with pytest.raises(ValidationError, match="at least one member"):
        CouncilConfig(enabled=True, members=[])


def test_build_council_context_uses_config_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_config(key: str | None = None) -> str | None:
        if key == "DEFAULT_LLM_MODEL":
            return "configured-model"
        return None

    monkeypatch.setattr(council_orchestrator, "get_config", fake_get_config)
    monkeypatch.setattr(
        council_orchestrator,
        "load_council_config",
        lambda: {"members": [{"member_id": "member-1", "max_tokens": 128}]},
    )

    context = council_orchestrator._build_council_context()

    assert context.member_model == "configured-model"
    assert context.judge_model == "configured-model"
    assert context.config.members[0].model == "configured-model"


def test_build_council_context_requires_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(council_orchestrator, "get_config", lambda *_: None)
    monkeypatch.setattr(
        council_orchestrator,
        "load_council_config",
        lambda: {"members": [{"member_id": "member-1", "max_tokens": 128}]},
    )

    with pytest.raises(RuntimeError, match="DEFAULT_LLM_MODEL must be configured"):
        council_orchestrator._build_council_context()


def test_council_orchestrator_peer_vote_mode() -> None:
    orchestrator = CouncilOrchestrator()
    config = _build_council_config(voting_mode="peer_vote")
    question = _build_question()

    outcome = orchestrator.deliberate(config, question)

    assert outcome.winner_id == "facilitator"
    assert outcome.winning_member_ids == ["facilitator"]
    assert outcome.resolution == "facilitator proposal selected"
    assert outcome.votes["facilitator"] == pytest.approx(2.7)
    assert outcome.summary == "Peer vote aggregation complete."
    assert outcome.metadata is not None
    metrics = outcome.metadata.get("metrics", {})
    assert metrics["peer_vote"]["voter_count"] == len(config.members)
    assert "facilitator" in metrics["peer_vote"]["votes"]["facilitator"]


def test_council_orchestrator_heuristic_mode() -> None:
    orchestrator = CouncilOrchestrator()
    config = _build_council_config(voting_mode="heuristic")
    question = _build_question()

    outcome = orchestrator.deliberate(config, question)

    assert outcome.winner_id == "facilitator"
    assert outcome.votes["facilitator"] == pytest.approx(0.82)
    assert outcome.summary == "Heuristic scoring applied to council answers."
    assert outcome.metadata is not None
    metrics = outcome.metadata.get("metrics", {})
    assert metrics["heuristic"]["scores"]["facilitator"] == pytest.approx(0.82)
