import importlib
from unittest.mock import MagicMock

import pytest

from src.agents.core.agent_state import AgentState
from src.infra import llm_client as llm_client_mod
from src.infra.config import get_config


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
