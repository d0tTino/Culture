from __future__ import annotations

# ruff: noqa: ANN101
import logging
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from .settings import settings

# Skip self argument annotation warnings for class methods


class Ledger:
    """Simple SQLite-backed ledger tracking agent resources."""

    def __init__(self, db_path: str | Path = "ledger.sqlite3") -> None:
        path = Path(db_path)
        self.conn = sqlite3.connect(path.as_posix(), timeout=60.0, check_same_thread=False)
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
                gas_price_per_call REAL DEFAULT 0,
                gas_price_per_token REAL DEFAULT 0,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_tokens (
                agent_id TEXT,
                token TEXT,
                amount INTEGER DEFAULT 0,
                PRIMARY KEY(agent_id, token)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_stakes (
                action_id TEXT,
                agent_id TEXT,
                amount REAL,
                PRIMARY KEY(action_id, agent_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auctions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item TEXT,
                status TEXT DEFAULT 'open',
                winner_id TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                auction_id INTEGER,
                agent_id TEXT,
                amount REAL,
                FOREIGN KEY(auction_id) REFERENCES auctions(id) ON DELETE CASCADE
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS genealogy (
                parent_id TEXT,
                child_id TEXT PRIMARY KEY
            )
            """
        )
        self.conn.commit()
        self.gas_price_per_call = float(settings.GAS_PRICE_PER_CALL)
        self.gas_price_per_token = float(settings.GAS_PRICE_PER_TOKEN)
        self._hooks: list[Callable[[str, float, float, str, float, float], None]] = []
        self.register_hook(self._db_hook)

    def register_hook(self, hook: Callable[[str, float, float, str, float, float], None]) -> None:
        """Register a hook called for every ``log_change`` invocation."""
        self._hooks.append(hook)

    def _db_hook(
        self,
        agent_id: str,
        delta_ip: float,
        delta_du: float,
        reason: str,
        gas_price_per_call: float,
        gas_price_per_token: float,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO transactions(agent_id, delta_ip, delta_du, reason, gas_price_per_call, gas_price_per_token) VALUES (?,?,?,?,?,?)",
            (
                agent_id,
                float(delta_ip),
                float(delta_du),
                reason,
                float(gas_price_per_call),
                float(gas_price_per_token),
            ),
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

    def log_change(
        self,
        agent_id: str,
        delta_ip: float = 0.0,
        delta_du: float = 0.0,
        reason: str = "",
        gas_price_per_call: float | None = None,
        gas_price_per_token: float | None = None,
    ) -> None:
        """Record a transaction and invoke registered hooks."""
        gpc = float(gas_price_per_call or 0.0)
        gpt = float(gas_price_per_token or 0.0)
        for hook in self._hooks:
            try:
                for _ in range(3):
                    try:
                        hook(
                            agent_id,
                            float(delta_ip),
                            float(delta_du),
                            reason,
                            gpc,
                            gpt,
                        )
                        break
                    except sqlite3.OperationalError:
                        time.sleep(0.05)
                else:
                    raise
            except Exception as e:  # pragma: no cover - defensive
                logging.getLogger(__name__).warning(
                    "Ledger hook failed for %s: %s",
                    agent_id,
                    e,
                    exc_info=True,
                )

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

    def stake_du_for_action(self, agent_id: str, action_id: str, amount: float) -> None:
        """Stake DU for a specific action."""
        if amount <= 0:
            return
        self.stake_du(agent_id, amount)
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO action_stakes(action_id, agent_id, amount)
            VALUES(?, ?, ?)
            ON CONFLICT(action_id, agent_id) DO UPDATE SET
                amount = amount + excluded.amount
            """,
            (action_id, agent_id, float(amount)),
        )
        self.conn.commit()

    def claim_action_refund(self, agent_id: str, action_id: str) -> None:
        """Refund staked DU for an action back to the agent."""
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT amount FROM action_stakes WHERE action_id=? AND agent_id=?",
            (action_id, agent_id),
        ).fetchone()
        if row:
            amt = float(row[0])
            self.unstake_du(agent_id, amt)
            cur.execute(
                "DELETE FROM action_stakes WHERE action_id=? AND agent_id=?",
                (action_id, agent_id),
            )
            self.conn.commit()

    def set_gas_prices(
        self, per_call: float | None = None, per_token: float | None = None
    ) -> None:
        """Update tracked gas price values."""
        if per_call is not None:
            self.gas_price_per_call = float(per_call)
        if per_token is not None:
            self.gas_price_per_token = float(per_token)

    def calculate_gas_price(self, agent_id: str, window: int = 10) -> tuple[float, float]:
        """Adjust gas prices based on recent DU burn rate."""
        burn_rate = self.get_du_burn_rate(agent_id, window)
        factor = 1.0 + burn_rate / 10.0
        base_call = float(settings.GAS_PRICE_PER_CALL)
        base_token = float(settings.GAS_PRICE_PER_TOKEN)
        new_call = base_call * factor
        new_token = base_token * factor
        changed = (
            abs(new_call - self.gas_price_per_call) > 1e-6
            or abs(new_token - self.gas_price_per_token) > 1e-6
        )
        if changed:
            self.set_gas_prices(new_call, new_token)
            self.log_change(
                agent_id,
                0.0,
                0.0,
                "gas_price_update",
                gas_price_per_call=new_call,
                gas_price_per_token=new_token,
            )
            try:  # pragma: no cover - optional dependency
                from src.interfaces import metrics

                metrics.GAS_PRICE_PER_CALL.set(new_call)
                metrics.GAS_PRICE_PER_TOKEN.set(new_token)
            except Exception:
                pass
        return new_call, new_token

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

    def add_tokens(self, agent_id: str, token: str, amount: int) -> None:
        if amount <= 0:
            return
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO agent_tokens(agent_id, token, amount)
            VALUES(?, ?, ?)
            ON CONFLICT(agent_id, token) DO UPDATE SET
                amount = amount + excluded.amount
            """,
            (agent_id, token, int(amount)),
        )
        self.conn.commit()

    def remove_tokens(self, agent_id: str, token: str, amount: int) -> None:
        if amount <= 0:
            return
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE agent_tokens
            SET amount = MAX(amount - ?, 0)
            WHERE agent_id=? AND token=?
            """,
            (int(amount), agent_id, token),
        )
        self.conn.commit()

    def get_tokens(self, agent_id: str, token: str) -> int:
        cur = self.conn.execute(
            "SELECT amount FROM agent_tokens WHERE agent_id=? AND token=?",
            (agent_id, token),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def record_genealogy(self, parent_id: str, child_id: str) -> None:
        """Record a parent/child relationship."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO genealogy(parent_id, child_id) VALUES(?, ?)",
            (parent_id, child_id),
        )
        self.conn.commit()

    # -------------------------------------------------------------
    # Auction management
    # -------------------------------------------------------------

    def open_auction(self, item: str) -> int:
        """Create a new auction and return its ID."""
        cur = self.conn.cursor()
        cur.execute("INSERT INTO auctions(item) VALUES(?)", (item,))
        self.conn.commit()
        return int(cur.lastrowid)

    def place_bid(self, auction_id: int, agent_id: str, amount: float) -> None:
        """Place a bid by staking DU."""
        if amount <= 0:
            return
        self.stake_du(agent_id, amount)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO bids(auction_id, agent_id, amount) VALUES(?,?,?)",
            (auction_id, agent_id, float(amount)),
        )
        self.conn.commit()

    def resolve_auction(self, auction_id: int) -> tuple[str | None, float]:
        """Resolve an auction, returning the winning agent and amount."""
        cur = self.conn.cursor()
        bids = cur.execute(
            "SELECT id, agent_id, amount FROM bids WHERE auction_id=?",
            (auction_id,),
        ).fetchall()
        if not bids:
            cur.execute(
                "UPDATE auctions SET status='resolved' WHERE id=?",
                (auction_id,),
            )
            self.conn.commit()
            return None, 0.0

        bids.sort(key=lambda r: (-float(r[2]), r[0]))
        winner_row = bids[0]
        winner_id = str(winner_row[1])
        winning_amount = float(winner_row[2])

        for _, agent, amt in bids:
            if agent == winner_id:
                cur.execute(
                    "UPDATE agent_balances SET staked_du = MAX(staked_du - ?, 0) WHERE agent_id=?",
                    (float(amt), agent),
                )
            else:
                self.unstake_du(str(agent), float(amt))

        cur.execute(
            "UPDATE auctions SET status='resolved', winner_id=? WHERE id=?",
            (winner_id, auction_id),
        )
        self.conn.commit()
        return winner_id, winning_amount


ledger = Ledger()

__all__ = [
    "Ledger",
    "ledger",
]
