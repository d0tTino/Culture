import pytest

from src.infra import config


@pytest.mark.unit
def test_missing_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    config.load_config(validate_required=True)


@pytest.mark.unit
def test_missing_llm_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "model")
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    config.load_config(validate_required=True)


@pytest.mark.unit
def test_empty_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "")
    with pytest.raises(RuntimeError) as exc:
        config.load_config(validate_required=True)
    assert "MODEL_NAME" in str(exc.value)


@pytest.mark.unit
def test_empty_llm_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "model")
    monkeypatch.setenv("LLM_API_BASE", "")
    cfg = config.load_config(validate_required=True)
    assert cfg["LLM_API_BASE"] == "http://localhost:11434"
