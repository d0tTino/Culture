import sqlite3

import pytest

from src.infra.ledger import Ledger

pytestmark = pytest.mark.unit


def test_transaction_record_and_balance(tmp_path):
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


def test_unknown_agent_balance(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    assert ledger.get_balance("nope") == (0.0, 0.0)
