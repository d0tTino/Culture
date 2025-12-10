from __future__ import annotations

import json

import pytest

from src.agents.council import (
    CouncilConfig,
    CouncilMemberConfig,
    CouncilOrchestrator,
    CouncilQuestion,
)
from src.infra import llm_client
from src.shared import llm_mocks

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    llm_mocks.patch_ollama_functions(monkeypatch)
    llm_client.enable_mock_mode(
        True,
        {
            "CouncilVoteModel": {
                "winning_member_id": "facilitator",
                "winner": "facilitator",
                "votes": {"facilitator": 1, "innovator": 1, "analyst": 1},
                "scores": {"facilitator": 1.0, "innovator": 1.0, "analyst": 1.0},
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


def _build_council_config() -> CouncilConfig:
    return CouncilConfig(
        enabled=True,
        members=[
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
        ],
    )


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

    assert outcome.winning_member_ids == ["facilitator"]
    assert outcome.resolution == "facilitator proposal selected"
    assert outcome.metadata is not None
    metrics = outcome.metadata.get("metrics", {})
    assert metrics.get("du_budget_exhausted") is False
    assert metrics.get("du_budget_per_member") == pytest.approx(5.0)
    assert outcome.summary == "Deterministic council summary"

    serialized = json.loads(json.dumps(outcome.metadata))
    assert serialized["metrics"]["cohesion"] == pytest.approx(0.91)
