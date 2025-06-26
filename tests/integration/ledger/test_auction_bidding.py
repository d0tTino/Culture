from pathlib import Path

import pytest

from src.infra.ledger import Ledger


@pytest.mark.integration
def test_auction_bidding_and_resolution(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")

    ledger.log_change("A", 0.0, 10.0, "fund")
    ledger.log_change("B", 0.0, 8.0, "fund")

    auction_id = ledger.open_auction("test")
    ledger.place_bid(auction_id, "A", 5)
    ledger.place_bid(auction_id, "B", 7)

    winner, amount = ledger.resolve_auction(auction_id)
    assert winner == "B"
    assert amount == pytest.approx(7)

    _, du_a = ledger.get_balance("A")
    _, du_b = ledger.get_balance("B")

    assert du_a == pytest.approx(10.0)
    assert du_b == pytest.approx(1.0)
    assert ledger.get_staked_du("A") == 0.0
    assert ledger.get_staked_du("B") == 0.0


@pytest.mark.integration
def test_auction_no_bids(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")

    auction_id = ledger.open_auction("empty")
    winner, amount = ledger.resolve_auction(auction_id)

    assert winner is None
    assert amount == pytest.approx(0.0)


@pytest.mark.integration
def test_insufficient_balance_does_not_go_negative(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")

    ledger.log_change("A", 0.0, 5.0, "fund")
    ledger.log_change("A", 0.0, -10.0, "overspend")

    _, du = ledger.get_balance("A")
    assert du == pytest.approx(0.0)
