import importlib
from unittest.mock import MagicMock

import pytest

from src.infra import config, llm_client
from src.shared import decorator_utils


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

    monkeypatch.setattr(llm_client, "LLM_API_BASE", "http://ollama:1234")
    monkeypatch.setattr(llm_client, "VLLM_API_BASE", "http://vllm:8001")
    monkeypatch.setattr(llm_client, "USE_VLLM", True)
    monkeypatch.setattr(llm_client, "client", llm_client._create_vllm_client())  # type: ignore[attr-defined]
    monkeypatch.setattr(config.settings, "LLM_API_BASE", "http://ollama:1234")  # type: ignore[attr-defined]
    monkeypatch.setattr(config.settings, "VLLM_API_BASE", "http://vllm:8001")  # type: ignore[attr-defined]
    monkeypatch.setitem(config._CONFIG, "LLM_API_BASE", "http://ollama:1234")
    monkeypatch.setitem(config._CONFIG, "VLLM_API_BASE", "http://vllm:8001")
    monkeypatch.setattr(decorator_utils.llm_perf_logger, "info", lambda *a: None)
    monkeypatch.setattr(decorator_utils.json, "dumps", lambda *a, **k: "{}")  # type: ignore[attr-defined]
    monkeypatch.setattr(llm_client.requests, "post", fake_post)  # type: ignore[attr-defined]

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
    config.load_config(validate_required=False)
    module = importlib.reload(llm_client)
    monkeypatch.setattr(decorator_utils.llm_perf_logger, "info", lambda *a: None)
    monkeypatch.setattr(decorator_utils.json, "dumps", lambda *a, **k: "{}")  # type: ignore[attr-defined]
    monkeypatch.setattr(module, "LLM_API_BASE", "http://ollama:1234")
    monkeypatch.setattr(module, "VLLM_API_BASE", "http://vllm:8002")
    monkeypatch.setattr(module, "USE_VLLM", True)
    monkeypatch.setattr(module, "client", module._create_vllm_client())  # type: ignore[attr-defined]
    monkeypatch.setattr(config.settings, "LLM_API_BASE", "http://ollama:1234")  # type: ignore[attr-defined]
    monkeypatch.setattr(config.settings, "VLLM_API_BASE", "http://vllm:8002")  # type: ignore[attr-defined]
    monkeypatch.setitem(config._CONFIG, "LLM_API_BASE", "http://ollama:1234")
    monkeypatch.setitem(config._CONFIG, "VLLM_API_BASE", "http://vllm:8002")
    monkeypatch.setattr(module.requests, "post", fake_post)  # type: ignore[attr-defined]

    result = module.generate_text("hello")

    assert result == "hi"
    assert captured["url"] == "http://vllm:8002/v1/chat/completions"


@pytest.mark.unit
@pytest.mark.disable_global_llm_mock
def test_vllm_client_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``VLLM_API_BASE`` is unset, fallback to ``LLM_API_BASE``."""
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

    monkeypatch.setattr(llm_client, "LLM_API_BASE", "http://ollama:1234")
    monkeypatch.setattr(llm_client, "VLLM_API_BASE", "")
    monkeypatch.setattr(llm_client, "USE_VLLM", True)
    monkeypatch.setattr(llm_client, "client", llm_client._create_vllm_client())  # type: ignore[attr-defined]
    monkeypatch.setattr(decorator_utils.llm_perf_logger, "info", lambda *a: None)
    monkeypatch.setattr(decorator_utils.json, "dumps", lambda *a, **k: "{}")  # type: ignore[attr-defined]
    monkeypatch.setattr(config.settings, "LLM_API_BASE", "http://ollama:1234")  # type: ignore[attr-defined]
    monkeypatch.setattr(config.settings, "VLLM_API_BASE", "")  # type: ignore[attr-defined]
    monkeypatch.setitem(config._CONFIG, "LLM_API_BASE", "http://ollama:1234")
    monkeypatch.setitem(config._CONFIG, "VLLM_API_BASE", "")
    monkeypatch.setattr(llm_client.requests, "post", fake_post)  # type: ignore[attr-defined]

    result = llm_client.generate_text("hello")

    assert result == "hi"
    assert captured["url"] == "http://ollama:1234/v1/chat/completions"
