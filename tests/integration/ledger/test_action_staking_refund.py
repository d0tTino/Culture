from pathlib import Path

import pytest

from src.infra.ledger import Ledger


@pytest.mark.integration
def test_action_staking_and_refund(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")

    ledger.log_change("A", 0.0, 5.0, "fund")

    ledger.stake_du_for_action("A", "act1", 3.0)

    assert ledger.get_staked_du("A") == pytest.approx(3.0)
    _, du = ledger.get_balance("A")
    assert du == pytest.approx(2.0)

    ledger.claim_action_refund("A", "act1")

    assert ledger.get_staked_du("A") == pytest.approx(0.0)
    _, du_after = ledger.get_balance("A")
    assert du_after == pytest.approx(5.0)
