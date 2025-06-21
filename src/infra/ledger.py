from __future__ import annotations

import sqlite3
from pathlib import Path


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
                du REAL DEFAULT 0
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
            VALUES(?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                ip = ip + excluded.ip,
                du = du + excluded.du
            """,
            (agent_id, float(delta_ip), float(delta_du)),
        )
        self.conn.commit()

    def get_balance(self, agent_id: str) -> tuple[float, float]:
        """Return the current balance for ``agent_id``."""
        cur = self.conn.execute("SELECT ip, du FROM agent_balances WHERE agent_id=?", (agent_id,))
        row = cur.fetchone()
        if row:
            return float(row[0]), float(row[1])
        return 0.0, 0.0


ledger = Ledger()

__all__ = ["Ledger", "ledger"]
