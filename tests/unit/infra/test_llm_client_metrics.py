import importlib

import pytest

from src.infra import llm_client as llm_client_mod
from src.interfaces import metrics


@pytest.mark.unit
def test_llm_client_chat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(llm_client_mod)

    class DummyClient:
        def chat(self, model: str, messages: list[dict], options: dict | None = None) -> dict:
            return {"message": {"content": "ok"}}

    monkeypatch.setattr(module, "get_ollama_client", lambda: DummyClient())
    client = module.LLMClient(module.LLMClientConfig())

    before_calls = metrics.LLM_CALLS_TOTAL._value.get()
    before_errors = metrics.LLM_ERRORS_TOTAL._value.get()

    result = client.chat(model="mistral:latest", messages=[{"role": "user", "content": "hi"}])
    assert result == {"message": {"content": "ok"}}

    assert metrics.LLM_CALLS_TOTAL._value.get() == before_calls + 1
    assert metrics.LLM_ERRORS_TOTAL._value.get() == before_errors


@pytest.mark.unit
def test_llm_client_chat_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(llm_client_mod)

    class DummyClient:
        def chat(self, model: str, messages: list[dict], options: dict | None = None) -> dict:
            raise module.RequestException("boom")

    monkeypatch.setattr(module, "get_ollama_client", lambda: DummyClient())
    client = module.LLMClient(module.LLMClientConfig())

    before_calls = metrics.LLM_CALLS_TOTAL._value.get()
    before_errors = metrics.LLM_ERRORS_TOTAL._value.get()

    with pytest.raises(module.RequestException):
        client.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert metrics.LLM_CALLS_TOTAL._value.get() == before_calls + 1
    assert metrics.LLM_ERRORS_TOTAL._value.get() == before_errors + 1
