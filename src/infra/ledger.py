from __future__ import annotations

import sqlite3

# from Path is not typed for self methods
from pathlib import Path

# Skip self argument annotation warnings for class methods


class Ledger:
    """Simple SQLite-backed ledger tracking agent resources."""

    def __init__(self, db_path: str | Path = "ledger.sqlite3") -> None:
        path = Path(db_path)
        self.conn = sqlite3.connect(path.as_posix())
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_balances (
                agent_id TEXT PRIMARY KEY,
                ip REAL DEFAULT 0,
                du REAL DEFAULT 0,
                staked_du REAL DEFAULT 0
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                delta_ip REAL,
                delta_du REAL,
                reason TEXT,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def log_change(
        self, agent_id: str, delta_ip: float = 0.0, delta_du: float = 0.0, reason: str = ""
    ) -> None:
        """Record a transaction and update balances."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO transactions(agent_id, delta_ip, delta_du, reason) VALUES (?,?,?,?)",
            (agent_id, float(delta_ip), float(delta_du), reason),
        )
        cur.execute(
            """
            INSERT INTO agent_balances(agent_id, ip, du)
            VALUES(?, MAX(?, 0), MAX(?, 0))
            ON CONFLICT(agent_id) DO UPDATE SET
                ip = MAX(ip + ?, 0),
                du = MAX(du + ?, 0)
            """,
            (
                agent_id,
                float(delta_ip),
                float(delta_du),
                float(delta_ip),
                float(delta_du),
            ),
        )
        self.conn.commit()

    def stake_du(self, agent_id: str, amount: float) -> None:
        """Stake DU for ``agent_id`` and record the transaction."""
        if amount <= 0:
            return
        self.log_change(agent_id, 0.0, -amount, "stake")
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO agent_balances(agent_id, staked_du)
            VALUES(?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                staked_du = staked_du + excluded.staked_du
            """,
            (agent_id, float(amount)),
        )
        self.conn.commit()

    def unstake_du(self, agent_id: str, amount: float) -> None:
        """Unstake DU for ``agent_id`` and record the transaction."""
        if amount <= 0:
            return
        self.log_change(agent_id, 0.0, amount, "unstake")
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE agent_balances SET staked_du = MAX(staked_du - ?, 0) WHERE agent_id=?",
            (float(amount), agent_id),
        )
        self.conn.commit()

    def get_staked_du(self, agent_id: str) -> float:
        """Return the amount of staked DU for ``agent_id``."""
        cur = self.conn.execute(
            "SELECT staked_du FROM agent_balances WHERE agent_id=?",
            (agent_id,),
        )
        row = cur.fetchone()
        return float(row[0]) if row else 0.0

    def get_du_burn_rate(self, agent_id: str, window: int = 10) -> float:
        """Return the average DU spent over the last ``window`` transactions."""
        cur = self.conn.execute(
            "SELECT delta_du FROM transactions WHERE agent_id=? ORDER BY id DESC LIMIT ?",
            (agent_id, int(window)),
        )
        rows = cur.fetchall()
        spent = [-float(r[0]) for r in rows if r[0] < 0]
        if not spent:
            return 0.0
        return sum(spent) / len(spent)

    def get_balance(self, agent_id: str) -> tuple[float, float]:
        """Return the current balance for ``agent_id``."""
        cur = self.conn.execute("SELECT ip, du FROM agent_balances WHERE agent_id=?", (agent_id,))
        row = cur.fetchone()
        if row:
            return float(row[0]), float(row[1])
        return 0.0, 0.0


ledger = Ledger()

__all__ = [
    "Ledger",
    "ledger",
]
