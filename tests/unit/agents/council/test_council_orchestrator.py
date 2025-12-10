from __future__ import annotations

import json

import pytest

from src.agents.council import CouncilConfig, CouncilMemberConfig, CouncilOrchestrator, CouncilQuestion
from src.infra import llm_client
from src.shared import llm_mocks


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_mocks.patch_ollama_functions(monkeypatch)


def _build_council_config(num_members: int = 3) -> CouncilConfig:
    base_members = [
        CouncilMemberConfig(
            member_id="facilitator",
            display_name="Facilitator",
            role="Moderator",
            description="Ensures everyone is heard",
            system_prompt="Lead with clarity",
            decision_weight=1.0,
        ),
        CouncilMemberConfig(
            member_id="innovator",
            display_name="Innovator",
            role="Idea generator",
            description="Pushes creative thinking",
            system_prompt="Bring new ideas",
            decision_weight=1.0,
        ),
        CouncilMemberConfig(
            member_id="analyst",
            display_name="Analyst",
            role="Evaluator",
            description="Stress-tests ideas",
            system_prompt="Look for gaps",
            decision_weight=1.0,
        ),
    ]

    extra_members: list[CouncilMemberConfig] = []
    for idx in range(3, num_members):
        extra_members.append(
            CouncilMemberConfig(
                member_id=f"member-{idx}",
                display_name=f"Member {idx}",
                role="Specialist",
                description="Brings domain expertise",
                system_prompt="Share focused insight",
                decision_weight=1.0,
            )
        )

    return CouncilConfig(enabled=True, members=base_members + extra_members)


def _build_question() -> CouncilQuestion:
    return CouncilQuestion(
        question_id="q-1",
        prompt="Which project should we fund first?",
        context="We have budget for only one initiative this quarter.",
    )


def test_council_orchestrator_invokes_all_members_and_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    orchestrator = CouncilOrchestrator()
    config = _build_council_config()
    question = _build_question()

    outcome = orchestrator.deliberate(config, question)

    expected_member_ids = {member.member_id for member in config.members}
    observed_member_ids = {answer.member_id for answer in outcome.answers}

    assert observed_member_ids == expected_member_ids

    assert llm_client.client.generate.call_count == len(config.members) + 1

    assert outcome.winning_member_ids == ["facilitator"]
    assert outcome.resolution == "facilitator proposal selected"
    assert outcome.metadata == {
        "votes": {"facilitator": 1, "innovator": 1, "analyst": 1},
        "metrics": {"cohesion": 0.91, "coverage": 0.77},
    }
    assert outcome.summary == "Deterministic council summary"

    serialized = json.loads(json.dumps(outcome.metadata))
    assert serialized["metrics"]["cohesion"] == pytest.approx(0.91)


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
    assert outcome.metadata == {
        "du_exhausted": True,
        "partial": True,
        "completed_members": ["facilitator", "innovator"],
    }

    llm_mocks.set_mock_llm_du_budget(None)
