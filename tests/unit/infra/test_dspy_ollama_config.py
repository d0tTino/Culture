import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.infra import dspy_ollama_integration
from src.infra.dspy_ollama_integration import (
    OllamaLM,
    configure_dspy_with_ollama,
)


@pytest.mark.unit
def test_configure_dspy_with_ollama_server_unavailable(caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch_target = "src.infra.dspy_ollama_integration"
    with (
        patch(f"{monkeypatch_target}.DSPY_AVAILABLE", True),
        patch(f"{monkeypatch_target}.requests.get", side_effect=Exception("boom")),
        patch(f"{monkeypatch_target}.ollama.Client", MagicMock()),
    ):
        with caplog.at_level(logging.CRITICAL):
            result = configure_dspy_with_ollama(api_base="http://bad")
    assert result is None
    assert "Ollama server not accessible" in caplog.text


@pytest.mark.unit
def test_configure_dspy_with_ollama_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dspy_ollama_integration, "DSPY_AVAILABLE", True)
    dummy_response = MagicMock(status_code=200)
    monkeypatch.setattr(dspy_ollama_integration.requests, "get", lambda *a, **k: dummy_response)
    monkeypatch.setattr(dspy_ollama_integration.ollama, "Client", MagicMock())

    called: dict[str, OllamaLM] = {}

    def fake_configure(*, lm: OllamaLM, **_: object) -> None:
        called["lm"] = lm

    dummy_settings = SimpleNamespace(configure=fake_configure)
    monkeypatch.setattr(dspy_ollama_integration.dspy, "settings", dummy_settings)

    result = configure_dspy_with_ollama(model_name="mistral", api_base="http://good")

    assert isinstance(result, OllamaLM)
    assert called.get("lm") is result
