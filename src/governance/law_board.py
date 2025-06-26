from __future__ import annotations

import sqlite3
from pathlib import Path


class LawBoard:
    """SQLite-backed board storing passed laws."""

    def __init__(self, db_path: str | Path = "laws.sqlite3") -> None:
        path = Path(db_path)
        self.conn = sqlite3.connect(path.as_posix())
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS laws (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                passed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def add_law(self, text: str) -> None:
        self.conn.execute("INSERT INTO laws(text) VALUES (?)", (text,))
        self.conn.commit()

    def get_laws(self) -> list[str]:
        cur = self.conn.cursor()
        rows = cur.execute("SELECT text FROM laws ORDER BY id ASC").fetchall()
        return [str(r[0]) for r in rows]

    def close(self) -> None:
        self.conn.close()


law_board = LawBoard()

__all__ = ["LawBoard", "law_board"]
