"""Tests for council fitness metrics tracking."""

from __future__ import annotations

import pytest

from src.agents.council import CouncilFitnessStore
from src.agents.council.types import CouncilOutcome, CouncilQuestion, MemberAnswer

pytestmark = pytest.mark.unit


def _make_outcome(
    question_id: str,
    winners: list[str],
    agreement_score: float | None,
) -> CouncilOutcome:
    question = CouncilQuestion(question_id=question_id, prompt="Sample prompt")
    answers = [
        MemberAnswer(member_id="member-a", answer="Answer A"),
        MemberAnswer(member_id="member-b", answer="Answer B"),
        MemberAnswer(member_id="member-c", answer="Answer C"),
    ]
    metadata = {"agreement_score": agreement_score} if agreement_score is not None else None
    return CouncilOutcome(
        question=question,
        answers=answers,
        resolution="winner decided",
        winning_member_ids=winners,
        metadata=metadata,
    )


def test_council_fitness_store_tracks_wins_and_win_rate() -> None:
    store = CouncilFitnessStore()

    store.record(_make_outcome("q1", ["member-a"], agreement_score=0.5))

    member_a = store.get_member_fitness("member-a")
    member_b = store.get_member_fitness("member-b")
    assert member_a.wins == 1
    assert member_a.appearances == 1
    assert member_a.win_rate == 1.0
    assert member_b.wins == 0
    assert member_b.appearances == 1
    assert member_b.win_rate == 0.0
    assert store.average_agreement_score == 0.5

    store.record(_make_outcome("q2", ["member-b"], agreement_score=0.75))

    member_a = store.get_member_fitness("member-a")
    member_b = store.get_member_fitness("member-b")
    assert member_a.wins == 1
    assert member_a.appearances == 2
    assert member_a.win_rate == 0.5
    assert member_b.wins == 1
    assert member_b.appearances == 2
    assert member_b.win_rate == 0.5
    assert store.average_agreement_score == 0.625


def test_council_fitness_store_handles_multiple_winners() -> None:
    store = CouncilFitnessStore()

    store.record(_make_outcome("q1", ["member-a", "member-b"], agreement_score=0.9))
    member_a = store.get_member_fitness("member-a")
    member_b = store.get_member_fitness("member-b")
    member_c = store.get_member_fitness("member-c")

    assert member_a.wins == 1
    assert member_b.wins == 1
    assert member_c.wins == 0
    assert member_a.win_rate == 1.0
    assert member_b.win_rate == 1.0
    assert member_c.win_rate == 0.0
    assert store.average_agreement_score == 0.9

    store.record(_make_outcome("q2", ["member-c"], agreement_score=0.3))
    member_c = store.get_member_fitness("member-c")

    assert member_c.wins == 1
    assert member_c.appearances == 2
    assert member_c.win_rate == 0.5
    assert store.average_agreement_score == 0.6
