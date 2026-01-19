import sqlite3
from pathlib import Path

import pytest

from src.agents.council.stats_store import CouncilStatsStore
from src.agents.council.types import CouncilOutcome, CouncilQuestion, MemberAnswer

pytestmark = pytest.mark.unit


@pytest.fixture()
def store(tmp_path: Path) -> CouncilStatsStore:
    db_dir = tmp_path / "council"
    db_dir.mkdir(parents=True, exist_ok=True)
    return CouncilStatsStore(db_path=db_dir / "stats.sqlite3")


def _build_outcome(question_id: str = "q-1") -> CouncilOutcome:
    question = CouncilQuestion(question_id=question_id, prompt="What now?")
    answers = [
        MemberAnswer(member_id="alpha", answer="Option A", confidence=0.8),
        MemberAnswer(member_id="beta", answer="Option A", confidence=0.6),
        MemberAnswer(member_id="gamma", answer="Option B", confidence=0.4),
    ]
    return CouncilOutcome(
        question=question,
        answers=answers,
        resolution="Option A",
        winning_member_ids=["alpha"],
    )


def test_stats_store_records_member_and_pairwise_metrics(
    store: CouncilStatsStore,
) -> None:
    outcome = _build_outcome()

    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.5, "gamma": 0.0},
        ema_alpha=0.5,
    )

    alpha_stats = store.get_member_stats("alpha")
    assert alpha_stats["participations"] == 1
    assert alpha_stats["wins"] == 1
    assert alpha_stats["avg_confidence"] == pytest.approx(0.8)

    beta_stats = store.get_member_stats("beta")
    assert beta_stats["participations"] == 1
    assert beta_stats["wins"] == 0
    assert beta_stats["win_rate"] == 0.0

    pairwise = store.get_pairwise_agreement("alpha", "beta")
    assert pairwise["agreements"] == 1
    assert pairwise["disagreements"] == 0
    assert pairwise["ema_agreement"] == pytest.approx(0.5)
    assert pairwise["ema_last_updated"]
    assert pairwise["ema_high_agreement"] is False
    divergent = store.get_pairwise_agreement("alpha", "gamma")
    assert divergent["agreements"] == 0
    assert divergent["disagreements"] == 1


def test_stats_store_serializes_aggregates(store: CouncilStatsStore) -> None:
    store.record_batch([_build_outcome(), _build_outcome()])

    snapshot = store.serialize_metrics()

    assert snapshot["members"]
    alpha_entry = next(m for m in snapshot["members"] if m["member_id"] == "alpha")
    assert alpha_entry["participations"] == 2
    assert alpha_entry["wins"] == 2
    assert alpha_entry["win_rate"] == 1.0

    pairwise_entries = {(p["member_a"], p["member_b"]): p for p in snapshot["pairwise"]}
    assert pairwise_entries[("alpha", "beta")]["agreement_rate"] == 1.0
    assert pairwise_entries[("alpha", "beta")]["ema_agreement"] == 1.0
    assert pairwise_entries[("alpha", "beta")]["ema_high_agreement"] is True
    assert pairwise_entries[("alpha", "beta")]["ema_last_updated"]
    assert pairwise_entries[("alpha", "gamma")]["agreements"] == 0


def test_stats_store_serializes_filtered_metrics(store: CouncilStatsStore) -> None:
    store.record_outcome(_build_outcome("q-1"))
    store.record_outcome(_build_outcome("q-2"))

    snapshot = store.serialize_metrics()
    filtered = store.serialize_metrics(question_id="q-1")

    alpha_snapshot = next(m for m in snapshot["members"] if m["member_id"] == "alpha")
    alpha_filtered = next(m for m in filtered["members"] if m["member_id"] == "alpha")

    assert alpha_snapshot["participations"] == 2
    assert alpha_filtered["participations"] == 1
    assert alpha_filtered["wins"] == 1

    pairwise_global = next(
        p for p in snapshot["pairwise"] if p["member_a"] == "alpha" and p["member_b"] == "beta"
    )
    pairwise_filtered = next(
        p for p in filtered["pairwise"] if p["member_a"] == "alpha" and p["member_b"] == "beta"
    )
    assert pairwise_global["agreements"] == 2
    assert pairwise_filtered["agreements"] == 1


def test_stats_store_updates_pairwise_ema(store: CouncilStatsStore) -> None:
    outcome = _build_outcome()

    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.5, "gamma": 0.0},
        ema_alpha=0.5,
    )
    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
        ema_alpha=0.5,
    )

    pairwise = store.get_pairwise_agreement("alpha", "beta")
    assert pairwise["ema_agreement"] == pytest.approx(0.75)


def test_stats_store_updates_pairwise_ema_across_outcomes(
    store: CouncilStatsStore,
) -> None:
    outcome = _build_outcome()

    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
        ema_alpha=0.5,
    )
    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        ema_alpha=0.5,
    )
    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.8, "gamma": 0.0},
        ema_alpha=0.5,
    )

    pairwise = store.get_pairwise_agreement("alpha", "beta")
    assert pairwise["ema_agreement"] == pytest.approx(0.65)


def test_stats_store_serializes_ema_flags(store: CouncilStatsStore) -> None:
    outcome = _build_outcome()

    store.record_outcome(
        outcome,
        score_map={"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        ema_alpha=0.5,
    )

    snapshot = store.serialize_metrics()

    pairwise_entries = {(p["member_a"], p["member_b"]): p for p in snapshot["pairwise"]}
    alpha_beta = pairwise_entries[("alpha", "beta")]
    assert alpha_beta["ema_agreement"] == pytest.approx(0.0)
    assert alpha_beta["ema_high_agreement"] is False
    assert alpha_beta["ema_low_agreement"] is True
    assert alpha_beta["ema_last_updated"]


def test_stats_store_migrates_schema(tmp_path: Path) -> None:
    db_dir = tmp_path / "council"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "legacy.sqlite3"

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE council_member_stats (
            member_id TEXT PRIMARY KEY,
            participations INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            total_confidence REAL DEFAULT 0.0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE council_pairwise_agreements (
            member_a TEXT,
            member_b TEXT,
            agreements INTEGER DEFAULT 0,
            disagreements INTEGER DEFAULT 0,
            PRIMARY KEY(member_a, member_b)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO council_member_stats(member_id, participations, wins, total_confidence)
        VALUES('alpha', 2, 1, 1.2)
        """
    )
    conn.execute(
        """
        INSERT INTO council_pairwise_agreements(member_a, member_b, agreements, disagreements)
        VALUES('alpha', 'beta', 3, 1)
        """
    )
    conn.commit()
    conn.close()

    store = CouncilStatsStore(db_path=db_path)

    member_columns = {
        row[1] for row in store.conn.execute("PRAGMA table_info(council_member_stats)")
    }
    pairwise_columns = {
        row[1] for row in store.conn.execute("PRAGMA table_info(council_pairwise_agreements)")
    }
    assert "question_id" in member_columns
    assert "question_id" in pairwise_columns

    stats = store.get_member_stats("alpha")
    assert stats["participations"] == 2

    pairwise = store.get_pairwise_agreement("alpha", "beta")
    assert pairwise["agreements"] == 3

    question_id = store.conn.execute(
        "SELECT question_id FROM council_member_stats WHERE member_id='alpha'"
    ).fetchone()
    assert question_id
    assert question_id[0] == ""
