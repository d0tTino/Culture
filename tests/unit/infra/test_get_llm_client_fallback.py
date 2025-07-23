import pytest

from src.infra import config, llm_client


@pytest.mark.unit
@pytest.mark.disable_global_llm_mock
def test_get_llm_client_fallback_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_llm_client should return an Ollama client when vLLM init fails."""
    sentinel = object()

    def boom() -> None:
        raise llm_client.APIError("boom")

    def fake_retry(func, *args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception as e:  # pragma: no cover - handled below
            return None, e

    monkeypatch.setattr(llm_client, "_create_vllm_client", boom)
    monkeypatch.setattr(llm_client, "_create_ollama_client", lambda: sentinel)
    monkeypatch.setattr(llm_client, "_retry_with_backoff", fake_retry)
    monkeypatch.setattr(llm_client, "client", None)
    monkeypatch.setattr(llm_client, "LLM_API_BASE", "http://ollama:1234")
    monkeypatch.setattr(llm_client, "VLLM_API_BASE", "http://vllm:8001")
    monkeypatch.setattr(llm_client, "USE_VLLM", True)
    monkeypatch.setattr(config.settings, "LLM_API_BASE", "http://ollama:1234", raising=False)
    monkeypatch.setattr(config.settings, "VLLM_API_BASE", "http://vllm:8001", raising=False)
    monkeypatch.setitem(config._CONFIG, "LLM_API_BASE", "http://ollama:1234")
    monkeypatch.setitem(config._CONFIG, "VLLM_API_BASE", "http://vllm:8001")

    client = llm_client.get_llm_client()

    assert client is sentinel
    assert llm_client.USE_VLLM is False
