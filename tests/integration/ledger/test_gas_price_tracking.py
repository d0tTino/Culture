from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.core.agent_state import AgentState
from src.infra import config, llm_client
from src.infra.ledger import Ledger


@pytest.mark.integration
@pytest.mark.require_ollama
def test_gas_price_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.skip("skip in CI")
    monkeypatch.setenv("GAS_PRICE_PER_CALL", "1.0")
    monkeypatch.setenv("GAS_PRICE_PER_TOKEN", "0.0")
    config.load_config(validate_required=False)

    ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr(llm_client, "ledger", ledger)

    def fake_chat(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "message": {"content": "ok"},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(llm_client, "client", SimpleNamespace(chat=fake_chat))

    state = AgentState(agent_id="A", name="Agent", ip=0.0, du=5.0)
    ledger.log_change(state.agent_id, 0.0, 5.0, "fund")

    llm_client.generate_text("hi", agent_state=state)

    row = ledger.conn.execute(
        "SELECT gas_price_per_call, gas_price_per_token FROM transactions WHERE reason='llm_gas' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row == (1.0, 0.0)

    monkeypatch.setenv("GAS_PRICE_PER_CALL", "2.0")
    monkeypatch.setenv("GAS_PRICE_PER_TOKEN", "0.5")
    config.load_config(validate_required=False)

    llm_client.generate_text("bye", agent_state=state)

    row = ledger.conn.execute(
        "SELECT gas_price_per_call, gas_price_per_token FROM transactions WHERE reason='llm_gas' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row == (2.2, 0.55)
