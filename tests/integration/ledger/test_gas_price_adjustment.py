from pathlib import Path

import pytest

from src.agents.core.agent_state import AgentState
from src.infra import config, llm_client
from src.infra.ledger import Ledger
from src.interfaces import metrics


@pytest.mark.integration
def test_calculate_gas_price_adjusts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GAS_PRICE_PER_CALL", "1.0")
    monkeypatch.setenv("GAS_PRICE_PER_TOKEN", "0.1")
    config.load_config(validate_required=False)

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr(llm_client, "ledger", ledger)

    state = AgentState(agent_id="A", name="Agent", ip=0.0, du=5.0)
    ledger.log_change(state.agent_id, 0.0, 5.0, "fund")

    # Simulate recent burn to trigger price increase
    ledger.log_change(state.agent_id, 0.0, -3.0, "spend")
    monkeypatch.setattr(ledger, "get_du_burn_rate", lambda aid, window=10: 3.0)

    call_price, token_price = ledger.calculate_gas_price(state.agent_id)

    assert call_price == pytest.approx(1.3)
    assert token_price == pytest.approx(0.13)

    row = ledger.conn.execute(
        "SELECT reason, gas_price_per_call, gas_price_per_token FROM transactions WHERE reason='gas_price_update' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row == ("gas_price_update", 1.3, 0.13)

    assert metrics.GAS_PRICE_PER_CALL._value.get() == pytest.approx(1.3)
    assert metrics.GAS_PRICE_PER_TOKEN._value.get() == pytest.approx(0.13)
