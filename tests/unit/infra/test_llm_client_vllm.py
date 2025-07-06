import importlib
from unittest.mock import MagicMock

import pytest

from src.infra import llm_client


@pytest.mark.unit
@pytest.mark.disable_global_llm_mock
def test_generate_text_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_post(url: str, *args: object, **kwargs: object) -> MagicMock:
        captured["url"] = url
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return resp

    monkeypatch.setattr(llm_client, "VLLM_API_BASE", "http://vllm:8001")
    monkeypatch.setattr(llm_client, "USE_VLLM", True)
    monkeypatch.setattr(llm_client, "client", llm_client._create_vllm_client())
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    result = llm_client.generate_text("hello")

    assert result == "hi"
    assert captured["url"] == "http://vllm:8001/v1/chat/completions"


@pytest.mark.unit
@pytest.mark.disable_global_llm_mock
def test_generate_text_vllm_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting ``VLLM_API_BASE`` should make the client use the vLLM endpoint."""
    captured: dict[str, str] = {}

    def fake_post(url: str, *args: object, **kwargs: object) -> MagicMock:
        captured["url"] = url
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return resp

    monkeypatch.setenv("VLLM_API_BASE", "http://vllm:8002")
    module = importlib.reload(llm_client)
    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.generate_text("hello")

    assert result == "hi"
    assert captured["url"] == "http://vllm:8002/v1/chat/completions"
