from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.agents.council.types import CouncilOutcome, MemberAnswer


class CouncilStatsStore:
    """Persist council member statistics and agreement metrics."""

    _DEFAULT_EMA_ALPHA = 0.3
    _EMA_HIGH_THRESHOLD = 0.75
    _EMA_LOW_THRESHOLD = 0.25

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
                    ema_agreement REAL DEFAULT 0.0,
                    last_updated TEXT,
                    PRIMARY KEY(member_a, member_b)
                )
                """
            )
            self._ensure_pairwise_columns()
            self.conn.commit()

    def _ensure_pairwise_columns(self) -> None:
        columns = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(council_pairwise_agreements)")
        }
        if "ema_agreement" not in columns:
            self.conn.execute(
                "ALTER TABLE council_pairwise_agreements ADD COLUMN ema_agreement REAL DEFAULT 0.0"
            )
        if "last_updated" not in columns:
            self.conn.execute(
                "ALTER TABLE council_pairwise_agreements ADD COLUMN last_updated TEXT"
            )

    def _normalize_answer(self, answer: str | None) -> str:
        return (answer or "").strip().lower()

    @staticmethod
    def _coerce_score(value: object) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _extract_score_value(self, value: object) -> float | None:
        if isinstance(value, Mapping):
            total = self._coerce_score(value.get("total"))
            if total is not None:
                return total
            numeric_values = [
                score
                for score in (self._coerce_score(item) for item in value.values())
                if score is not None
            ]
            if numeric_values:
                return sum(numeric_values) / len(numeric_values)
            return None
        return self._coerce_score(value)

    def _normalize_score_map(self, score_map: Mapping[str, object] | None) -> dict[str, float]:
        if not isinstance(score_map, Mapping):
            return {}
        normalized: dict[str, float] = {}
        for member_id, value in score_map.items():
            score = self._extract_score_value(value)
            if score is not None:
                normalized[str(member_id)] = score
        return normalized

    def _extract_scores_from_outcome(self, outcome: CouncilOutcome) -> Mapping[str, object] | None:
        if outcome.votes:
            return outcome.votes
        if outcome.metadata and isinstance(outcome.metadata.get("scores"), Mapping):
            return outcome.metadata["scores"]
        if outcome.metrics:
            member_scores = outcome.metrics.get("member_scores")
            if isinstance(member_scores, Mapping):
                return member_scores
            scores = outcome.metrics.get("scores")
            if isinstance(scores, Mapping):
                return scores
        return None

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
        self,
        cur: sqlite3.Cursor,
        answers: Sequence[MemberAnswer],
        score_map: Mapping[str, float],
        ema_alpha: float,
        timestamp: str,
    ) -> None:
        scores = dict(score_map)
        score_min = min(scores.values()) if scores else 0.0
        score_max = max(scores.values()) if scores else 0.0
        score_range = score_max - score_min
        for idx, left in enumerate(answers):
            for right in answers[idx + 1 :]:
                member_a, member_b = sorted((left.member_id, right.member_id))
                agrees = self._normalize_answer(left.answer) == self._normalize_answer(
                    right.answer
                )
                agreement_score: float
                if member_a in scores and member_b in scores:
                    delta = abs(scores[member_a] - scores[member_b])
                    if score_range:
                        agreement_score = 1.0 - (delta / score_range)
                    else:
                        agreement_score = 1.0
                    agreement_score = max(0.0, min(1.0, agreement_score))
                else:
                    agreement_score = 1.0 if agrees else 0.0
                cur.execute(
                    """
                    INSERT INTO council_pairwise_agreements(
                        member_a,
                        member_b,
                        agreements,
                        disagreements,
                        ema_agreement,
                        last_updated
                    )
                    VALUES(?, ?, ?, ?, 0.0, NULL)
                    ON CONFLICT(member_a, member_b) DO UPDATE SET
                        agreements = agreements + excluded.agreements,
                        disagreements = disagreements + excluded.disagreements
                    """,
                    (member_a, member_b, 1 if agrees else 0, 0 if agrees else 1),
                )
                row = cur.execute(
                    """
                    SELECT ema_agreement, last_updated
                    FROM council_pairwise_agreements
                    WHERE member_a=? AND member_b=?
                    """,
                    (member_a, member_b),
                ).fetchone()
                previous_ema = float(row[0]) if row and row[0] is not None else 0.0
                has_prior = bool(row and row[1])
                new_ema = (
                    agreement_score
                    if not has_prior
                    else (ema_alpha * agreement_score + (1.0 - ema_alpha) * previous_ema)
                )
                cur.execute(
                    """
                    UPDATE council_pairwise_agreements
                    SET ema_agreement=?, last_updated=?
                    WHERE member_a=? AND member_b=?
                    """,
                    (new_ema, timestamp, member_a, member_b),
                )

    def record_outcome(
        self,
        outcome: CouncilOutcome,
        *,
        score_map: Mapping[str, float] | None = None,
        ema_alpha: float | None = None,
    ) -> None:
        """Persist statistics for a council ``outcome``."""

        answers = list(outcome.answers)
        if not answers:
            return

        raw_scores = score_map or self._extract_scores_from_outcome(outcome)
        normalized_scores = self._normalize_score_map(raw_scores)
        alpha = self._DEFAULT_EMA_ALPHA if ema_alpha is None else float(ema_alpha)
        timestamp = datetime.now(timezone.utc).isoformat()
        winning_ids = {member_id for member_id in outcome.winning_member_ids}
        with self._lock:
            cur = self.conn.cursor()
            self._update_member_stats(cur, answers, winning_ids)
            self._update_pairwise_agreements(
                cur,
                answers,
                normalized_scores,
                alpha,
                timestamp,
            )
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

    def get_pairwise_agreement(
        self, member_a: str, member_b: str
    ) -> dict[str, float | str | bool]:
        left, right = sorted((member_a, member_b))
        with self._lock:
            row = self.conn.execute(
                """
                SELECT agreements, disagreements, ema_agreement, last_updated
                FROM council_pairwise_agreements
                WHERE member_a=? AND member_b=?
                """,
                (left, right),
            ).fetchone()
        if not row:
            return {
                "member_a": left,
                "member_b": right,
                "agreements": 0,
                "disagreements": 0,
                "agreement_rate": 0.0,
                "ema_agreement": 0.0,
                "ema_last_updated": None,
                "ema_high_agreement": False,
                "ema_low_agreement": False,
            }

        agreements, disagreements = int(row[0]), int(row[1])
        ema_agreement = float(row[2]) if row[2] is not None else 0.0
        last_updated = row[3]
        total = agreements + disagreements
        agreement_rate = agreements / total if total else 0.0
        return {
            "member_a": left,
            "member_b": right,
            "agreements": agreements,
            "disagreements": disagreements,
            "agreement_rate": agreement_rate,
            "ema_agreement": ema_agreement,
            "ema_last_updated": last_updated,
            "ema_high_agreement": ema_agreement >= self._EMA_HIGH_THRESHOLD,
            "ema_low_agreement": ema_agreement <= self._EMA_LOW_THRESHOLD,
        }

    def serialize_metrics(
        self, *, question_id: str | None = None
    ) -> dict[str, list[dict[str, float | str | bool]]]:
        with self._lock:
            member_rows = self.conn.execute(
                "SELECT member_id, participations, wins, total_confidence FROM council_member_stats"
            ).fetchall()
            pair_rows = self.conn.execute(
                """
                SELECT member_a, member_b, agreements, disagreements, ema_agreement, last_updated
                FROM council_pairwise_agreements
                """
            ).fetchall()

        members: list[dict[str, float | str | bool]] = []
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

        pairwise: list[dict[str, float | str | bool]] = []
        for member_a, member_b, agreements, disagreements, ema_agreement, last_updated in pair_rows:
            agreements_i = int(agreements)
            disagreements_i = int(disagreements)
            ema_value = float(ema_agreement) if ema_agreement is not None else 0.0
            total = agreements_i + disagreements_i
            pairwise.append(
                {
                    "member_a": str(member_a),
                    "member_b": str(member_b),
                    "agreements": agreements_i,
                    "disagreements": disagreements_i,
                    "agreement_rate": agreements_i / total if total else 0.0,
                    "ema_agreement": ema_value,
                    "ema_last_updated": last_updated,
                    "ema_high_agreement": ema_value >= self._EMA_HIGH_THRESHOLD,
                    "ema_low_agreement": ema_value <= self._EMA_LOW_THRESHOLD,
                }
            )

        # ``question_id`` is accepted for forward compatibility with question-scoped
        # metrics but currently returns global aggregates only.
        return {"members": members, "pairwise": pairwise}

    async def serialize_metrics_async(
        self, *, question_id: str | None = None
    ) -> dict[str, list[dict[str, float | str | bool]]]:
        return await asyncio.to_thread(self.serialize_metrics, question_id=question_id)

    def record_batch(self, outcomes: Iterable[CouncilOutcome]) -> None:
        """Convenience helper to persist multiple outcomes."""

        for outcome in outcomes:
            self.record_outcome(outcome)


council_stats_store = CouncilStatsStore()

__all__ = ["CouncilStatsStore", "council_stats_store"]
