import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.core.agent_state import AgentState
from src.infra import llm_client as llm_client_mod
from src.infra.config import get_config
from tests.utils.mock_llm import MockLLM


@pytest.mark.unit
def test_du_decreases_after_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(llm_client_mod)
    state = AgentState(agent_id="A", name="Agent")
    start_du = state.du

    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "hi"}}
    monkeypatch.setattr(module, "get_ollama_client", lambda: fake_client)
    monkeypatch.setattr(
        module,
        "_retry_with_backoff",
        lambda func, *a, **kw: (func(), None),
    )

    result = module.generate_text("hi", agent_state=state)

    assert result == "hi"
    expected = start_du - (get_config("GAS_PRICE_PER_CALL") + get_config("GAS_PRICE_PER_TOKEN"))
    assert state.du == pytest.approx(expected)


@pytest.mark.unit
def test_du_and_ledger_with_mockllm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = importlib.reload(llm_client_mod)
    orig_generate_text = module.generate_text
    state = AgentState(agent_id="A", name="Agent")
    start_du = state.du

    from src.infra.ledger import Ledger

    test_ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr(module, "ledger", test_ledger)

    with MockLLM({"default": "hi"}):
        result = orig_generate_text("hi", agent_state=state)

    assert result == "hi"
    expected = start_du - (get_config("GAS_PRICE_PER_CALL") + get_config("GAS_PRICE_PER_TOKEN"))
    assert state.du == pytest.approx(expected)
    row = test_ledger.conn.execute(
        "SELECT delta_du, reason FROM transactions WHERE agent_id=?",
        (state.agent_id,),
    ).fetchone()
    assert row is not None and row[0] < 0 and row[1] == "llm_gas"


@pytest.mark.unit
def test_du_never_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(llm_client_mod)
    state = AgentState(agent_id="A", name="Agent")
    state.du = 0.5

    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "hi"}}
    monkeypatch.setattr(module, "get_ollama_client", lambda: fake_client)
    monkeypatch.setattr(
        module,
        "_retry_with_backoff",
        lambda func, *a, **kw: (func(), None),
    )

    log_called = False

    def fake_log(agent_id: str, delta_ip: float, delta_du: float, reason: str) -> None:
        nonlocal log_called
        log_called = True

    monkeypatch.setattr(module.ledger, "log_change", fake_log)

    result = module.generate_text("hi", agent_state=state)

    assert result == "hi"
    assert state.du == pytest.approx(0.5)
    assert not log_called
