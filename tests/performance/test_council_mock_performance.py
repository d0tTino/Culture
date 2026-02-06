import time
from unittest.mock import patch

import pytest

from src.agents.council import orchestrator as council_orchestrator
from src.agents.council.orchestrator import CouncilVoteModel, MemberAnswer
from src.agents.council.types import CouncilConfig, CouncilMemberConfig, CouncilQuestion
from src.infra import config as infra_config


@pytest.mark.performance
@pytest.mark.parametrize("member_count", [3])
def test_council_run_meets_latency_and_budget(member_count: int) -> None:
    infra_config._CONFIG = {
        "USE_COUNCIL_MODE": True,
        "DEFAULT_LLM_MODEL": "http://localhost/mock",
    }
    members = [
        CouncilMemberConfig(
            member_id=f"member-{idx}",
            display_name=f"Member {idx}",
            persona="Moves quickly and keeps answers minimal.",
            model="mock-model",
            temperature=0.1,
            max_tokens=64,
            role="Generalist",
            is_active=True,
        )
        for idx in range(member_count)
    ]
    config = CouncilConfig(
        members=members,
        max_concurrent_calls=member_count,
        du_budget_per_question=2.0,
    )
    question = CouncilQuestion(question_id="perf-smoke", prompt="How fast is the mock council?")

    def _fake_member_call(*_: object, **__: object) -> MemberAnswer:
        return MemberAnswer(
            member_id=members[0].member_id,
            answer="mock-answer",
            reasoning="short",
            confidence=0.9,
            citations=[],
            metadata={"du_consumed": 0.05},
        )

    def _fake_judge(
        *_: object, answers: list[MemberAnswer] | None = None, **__: object
    ) -> CouncilVoteModel:
        winner_id = (answers or members)[0].member_id
        return CouncilVoteModel(
            winning_member_id=winner_id,
            votes={winner_id: 1.0},
            scores={winner_id: {"overall": 1.0}},
            metrics={"member_scores": {winner_id: {"correctness": 1.0}}},
            summary="mock-win",
            reasoning="",
            resolution="mock resolution",
        )

    orchestrator = council_orchestrator.CouncilOrchestrator(
        max_concurrent_calls=member_count, du_budget_per_question=2.0
    )

    with (
        patch.object(council_orchestrator, "_ask_council_member", side_effect=_fake_member_call),
        patch.object(council_orchestrator, "_judge_council_answers", side_effect=_fake_judge),
    ):
        start = time.perf_counter()
        outcome = orchestrator.deliberate(
            config,
            question,
            extra_context={"text": "throughput check"},
            allow_disabled_mode=True,
        )
        duration = time.perf_counter() - start

    assert duration < 1.5
    throughput_per_minute = 60.0 / duration
    assert throughput_per_minute >= 25

    metrics = outcome.metadata.get("metrics", {}) if outcome.metadata else {}
    per_member_budget = metrics.get("du_budget_per_member", {})
    assert per_member_budget
    assert all(value <= 2.0 for value in per_member_budget.values())
    assert not metrics.get("du_budget_exhausted")
    assert outcome.winner_id in {member.member_id for member in members}
