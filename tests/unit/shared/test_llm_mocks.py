import json
from unittest.mock import MagicMock

import pytest

from src.shared import llm_mocks


@pytest.mark.unit
def test_is_ollama_running_false(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySocket:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass
        def settimeout(self, *args: object) -> None:  # noqa: D401 - thin wrapper
            pass
        def connect(self, addr: tuple[str, int]) -> None:
            raise OSError()
        def close(self) -> None:
            pass

    monkeypatch.setattr("socket.socket", lambda *a, **k: DummySocket())
    assert llm_mocks.is_ollama_running() is False


@pytest.mark.unit
def test_create_mock_ollama_client_chat_and_generate() -> None:
    client = llm_mocks.create_mock_ollama_client()

    neg = client.chat(messages=[{"content": "Analyze the sentiment of the following message. Strongly disagree"}])
    assert json.loads(neg["message"]["content"])["sentiment_score"] == -0.7

    gen = client.generate(prompt="Your output fields are: `l1_summary` (str)\nrecent_events")
    assert "l1_summary" in json.loads(gen["response"])

    fallback = client.generate(prompt="something else")
    assert "Fallback" in json.loads(fallback["response"])["detail"]


@pytest.mark.unit
def test_patch_ollama_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.infra import llm_client

    llm_mocks.patch_ollama_functions(monkeypatch)
    assert llm_client.generate_text("test") == llm_mocks.mock_text_global
    assert isinstance(llm_client.client.chat, MagicMock)
