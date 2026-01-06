from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.agents.council.types import CouncilOutcome, MemberAnswer


class CouncilStatsStore:
    """Persist council member statistics and agreement metrics."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        resolved_path = Path(db_path or self._infer_default_path())
        self.conn = sqlite3.connect(
            resolved_path.as_posix(), timeout=60.0, check_same_thread=False
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _infer_default_path(self) -> Path:
        """Infer the ledger database path when none is provided."""

        try:  # pragma: no cover - defensive fallback
            from src.infra.ledger import ledger

            path = self._extract_db_path(ledger.conn)
            if path:
                return path
        except Exception:
            pass
        return Path("ledger.sqlite3")

    def _extract_db_path(self, connection: sqlite3.Connection) -> Path | None:
        row = connection.execute("PRAGMA database_list").fetchone()
        if row and row[2]:
            return Path(str(row[2]))
        return None

    def _ensure_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS council_member_stats (
                    member_id TEXT PRIMARY KEY,
                    participations INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    total_confidence REAL DEFAULT 0.0
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS council_pairwise_agreements (
                    member_a TEXT,
                    member_b TEXT,
                    agreements INTEGER DEFAULT 0,
                    disagreements INTEGER DEFAULT 0,
                    PRIMARY KEY(member_a, member_b)
                )
                """
            )
            self.conn.commit()

    def _normalize_answer(self, answer: str | None) -> str:
        return (answer or "").strip().lower()

    def _update_member_stats(
        self, cur: sqlite3.Cursor, answers: Sequence[MemberAnswer], winning_ids: set[str]
    ) -> None:
        for answer in answers:
            confidence = float(answer.confidence or 0.0)
            cur.execute(
                """
                INSERT INTO council_member_stats(member_id, participations, wins, total_confidence)
                VALUES(?, 1, ?, ?)
                ON CONFLICT(member_id) DO UPDATE SET
                    participations = participations + 1,
                    wins = wins + excluded.wins,
                    total_confidence = total_confidence + excluded.total_confidence
                """,
                (
                    answer.member_id,
                    1 if answer.member_id in winning_ids else 0,
                    confidence,
                ),
            )

    def _update_pairwise_agreements(
        self, cur: sqlite3.Cursor, answers: Sequence[MemberAnswer]
    ) -> None:
        for idx, left in enumerate(answers):
            for right in answers[idx + 1 :]:
                member_a, member_b = sorted((left.member_id, right.member_id))
                agrees = self._normalize_answer(left.answer) == self._normalize_answer(
                    right.answer
                )
                cur.execute(
                    """
                    INSERT INTO council_pairwise_agreements(member_a, member_b, agreements, disagreements)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(member_a, member_b) DO UPDATE SET
                        agreements = agreements + excluded.agreements,
                        disagreements = disagreements + excluded.disagreements
                    """,
                    (member_a, member_b, 1 if agrees else 0, 0 if agrees else 1),
                )

    def record_outcome(self, outcome: CouncilOutcome) -> None:
        """Persist statistics for a council ``outcome``."""

        answers = list(outcome.answers)
        if not answers:
            return

        winning_ids = {member_id for member_id in outcome.winning_member_ids}
        with self._lock:
            cur = self.conn.cursor()
            self._update_member_stats(cur, answers, winning_ids)
            self._update_pairwise_agreements(cur, answers)
            self.conn.commit()

    async def record_outcome_async(self, outcome: CouncilOutcome) -> None:
        await asyncio.to_thread(self.record_outcome, outcome)

    def get_member_stats(self, member_id: str) -> dict[str, float]:
        with self._lock:
            row = self.conn.execute(
                "SELECT participations, wins, total_confidence FROM council_member_stats WHERE member_id=?",
                (member_id,),
            ).fetchone()
        if not row:
            return {"member_id": member_id, "participations": 0, "wins": 0, "win_rate": 0.0, "avg_confidence": 0.0}

        participations, wins, total_confidence = int(row[0]), int(row[1]), float(row[2])
        win_rate = wins / participations if participations else 0.0
        avg_confidence = total_confidence / participations if participations else 0.0
        return {
            "member_id": member_id,
            "participations": participations,
            "wins": wins,
            "win_rate": win_rate,
            "avg_confidence": avg_confidence,
        }

    def get_pairwise_agreement(self, member_a: str, member_b: str) -> dict[str, float]:
        left, right = sorted((member_a, member_b))
        with self._lock:
            row = self.conn.execute(
                "SELECT agreements, disagreements FROM council_pairwise_agreements WHERE member_a=? AND member_b=?",
                (left, right),
            ).fetchone()
        if not row:
            return {
                "member_a": left,
                "member_b": right,
                "agreements": 0,
                "disagreements": 0,
                "agreement_rate": 0.0,
            }

        agreements, disagreements = int(row[0]), int(row[1])
        total = agreements + disagreements
        agreement_rate = agreements / total if total else 0.0
        return {
            "member_a": left,
            "member_b": right,
            "agreements": agreements,
            "disagreements": disagreements,
            "agreement_rate": agreement_rate,
        }

    def serialize_metrics(self, *, question_id: str | None = None) -> dict[str, list[dict[str, float]]]:
        with self._lock:
            member_rows = self.conn.execute(
                "SELECT member_id, participations, wins, total_confidence FROM council_member_stats"
            ).fetchall()
            pair_rows = self.conn.execute(
                "SELECT member_a, member_b, agreements, disagreements FROM council_pairwise_agreements"
            ).fetchall()

        members: list[dict[str, float]] = []
        for member_id, participations, wins, total_confidence in member_rows:
            participations_i = int(participations)
            wins_i = int(wins)
            total_conf = float(total_confidence)
            members.append(
                {
                    "member_id": str(member_id),
                    "participations": participations_i,
                    "wins": wins_i,
                    "win_rate": wins_i / participations_i if participations_i else 0.0,
                    "avg_confidence": total_conf / participations_i if participations_i else 0.0,
                }
            )

        pairwise: list[dict[str, float]] = []
        for member_a, member_b, agreements, disagreements in pair_rows:
            agreements_i = int(agreements)
            disagreements_i = int(disagreements)
            total = agreements_i + disagreements_i
            pairwise.append(
                {
                    "member_a": str(member_a),
                    "member_b": str(member_b),
                    "agreements": agreements_i,
                    "disagreements": disagreements_i,
                    "agreement_rate": agreements_i / total if total else 0.0,
                }
            )

        # ``question_id`` is accepted for forward compatibility with question-scoped
        # metrics but currently returns global aggregates only.
        return {"members": members, "pairwise": pairwise}

    async def serialize_metrics_async(self, *, question_id: str | None = None) -> dict[str, list[dict[str, float]]]:
        return await asyncio.to_thread(self.serialize_metrics, question_id=question_id)

    def record_batch(self, outcomes: Iterable[CouncilOutcome]) -> None:
        """Convenience helper to persist multiple outcomes."""

        for outcome in outcomes:
            self.record_outcome(outcome)


council_stats_store = CouncilStatsStore()

__all__ = ["CouncilStatsStore", "council_stats_store"]
