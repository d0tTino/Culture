import sqlite3
from pathlib import Path

import pytest

from src.infra.ledger import Ledger

pytestmark = pytest.mark.unit


def test_transaction_record_and_balance(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    ledger = Ledger(db)
    ledger.log_change("a1", 5.0, 2.0, "init")
    ledger.log_change("a1", -1.0, 0.0, "spend")

    ip, du = ledger.get_balance("a1")
    assert ip == pytest.approx(4.0)
    assert du == pytest.approx(2.0)

    conn = sqlite3.connect(db)
    row = conn.execute("SELECT COUNT(*) FROM transactions WHERE agent_id='a1'").fetchone()
    assert row[0] == 2
    conn.close()


def test_unknown_agent_balance(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    assert ledger.get_balance("nope") == (0.0, 0.0)


def test_staking_and_burn_rate(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    ledger = Ledger(db)
    ledger.log_change("a", 0.0, 10.0, "init")
    ledger.stake_du("a", 5.0)
    ip, du = ledger.get_balance("a")
    assert du == pytest.approx(5.0)
    assert ledger.get_staked_du("a") == pytest.approx(5.0)
    ledger.unstake_du("a", 2.0)
    ip, du = ledger.get_balance("a")
    assert du == pytest.approx(7.0)
    assert ledger.get_staked_du("a") == pytest.approx(3.0)
    for _ in range(3):
        ledger.log_change("a", 0.0, -1.0, "spend")
    rate = ledger.get_du_burn_rate("a", window=3)
    assert rate == pytest.approx(1.0)


def test_stake_unstake_noop_and_zero_burn_rate(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger.log_change("a", 0.0, 5.0, "init")
    ledger.stake_du("a", 0)
    ledger.stake_du("a", -1)
    assert ledger.get_staked_du("a") == pytest.approx(0.0)
    ip, du = ledger.get_balance("a")
    assert du == pytest.approx(5.0)

    ledger.unstake_du("a", 0)
    ledger.unstake_du("a", -2)
    assert ledger.get_staked_du("a") == pytest.approx(0.0)
    ip, du = ledger.get_balance("a")
    assert du == pytest.approx(5.0)

    assert ledger.get_du_burn_rate("a") == 0.0


def test_log_change_triggers_hooks(tmp_path: Path) -> None:
    db = tmp_path / "ledger.sqlite"
    ledger = Ledger(db)

    calls: list[tuple[str, float, float, str, float, float]] = []

    def hook(agent_id: str, dip: float, ddu: float, reason: str, gpc: float, gpt: float) -> None:
        calls.append((agent_id, dip, ddu, reason, gpc, gpt))

    ledger.register_hook(hook)
    ledger.log_change("a1", 1.0, 2.0, "test")

    assert calls == [("a1", 1.0, 2.0, "test", 0.0, 0.0)]
    ip, du = ledger.get_balance("a1")
    assert ip == pytest.approx(1.0)
    assert du == pytest.approx(2.0)
