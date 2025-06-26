from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src.infra.ledger import Ledger

# Provide a minimal stub for src.infra.llm_client so that tests can run
stub = types.ModuleType("llm_client")


class LLMClient:  # pragma: no cover - simple stub
    pass


stub.LLMClient = LLMClient
stub.client = object()
stub.get_ollama_client = lambda: stub.client
stub.generate_text = lambda *args, **kwargs: ""
stub.summarize_memory_context = lambda *args, **kwargs: ""
sys.modules.setdefault("src.infra.llm_client", stub)

# Basic stub for the external ``ollama`` package used by llm mocks
ollama_stub = types.ModuleType("ollama")
ollama_stub.Client = lambda *args, **kwargs: object()
ollama_stub.list = lambda *args, **kwargs: []
ollama_stub.pull = lambda *args, **kwargs: None
ollama_stub.show = lambda *args, **kwargs: {}
ollama_stub.chat = lambda *args, **kwargs: {}
ollama_stub.generate = lambda *args, **kwargs: {}
sys.modules.setdefault("ollama", ollama_stub)


@pytest.mark.integration
def test_open_auction_and_place_bid(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")

    ledger.log_change("A", 0.0, 10.0, "fund")

    auction_id = ledger.open_auction("collectible")
    assert isinstance(auction_id, int)

    row = ledger.conn.execute(
        "SELECT item, status FROM auctions WHERE id=?",
        (auction_id,),
    ).fetchone()
    assert row == ("collectible", "open")

    ledger.place_bid(auction_id, "A", 4.0)

    bid_row = ledger.conn.execute(
        "SELECT agent_id, amount FROM bids WHERE auction_id=?",
        (auction_id,),
    ).fetchone()
    assert bid_row == ("A", 4.0)

    assert ledger.get_staked_du("A") == pytest.approx(4.0)
    _, du = ledger.get_balance("A")
    assert du == pytest.approx(6.0)


@pytest.mark.integration
def test_resolve_auction_tie_breaking_and_refunds(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    for agent in ["A", "B", "C"]:
        ledger.log_change(agent, 0.0, 10.0, "fund")

    auction_id = ledger.open_auction("rare")
    ledger.place_bid(auction_id, "B", 7)
    ledger.place_bid(auction_id, "C", 7)
    ledger.place_bid(auction_id, "A", 5)

    winner, amount = ledger.resolve_auction(auction_id)
    assert winner == "B"
    assert amount == pytest.approx(7)

    row = ledger.conn.execute(
        "SELECT status, winner_id FROM auctions WHERE id=?",
        (auction_id,),
    ).fetchone()
    assert row == ("resolved", "B")

    _, du_a = ledger.get_balance("A")
    _, du_b = ledger.get_balance("B")
    _, du_c = ledger.get_balance("C")

    assert du_a == pytest.approx(10.0)
    assert du_b == pytest.approx(3.0)
    assert du_c == pytest.approx(10.0)

    assert ledger.get_staked_du("A") == 0.0
    assert ledger.get_staked_du("B") == 0.0
    assert ledger.get_staked_du("C") == 0.0

@pytest.mark.integration
def test_resolve_auction_no_bids(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")

    auction_id = ledger.open_auction("empty")
    winner, amount = ledger.resolve_auction(auction_id)

    assert winner is None
    assert amount == pytest.approx(0.0)

    row = ledger.conn.execute(
        "SELECT status, winner_id FROM auctions WHERE id=?",
        (auction_id,),
    ).fetchone()
    assert row == ("resolved", None)

