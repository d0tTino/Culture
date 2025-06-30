import pytest

from src.infra import config


@pytest.mark.unit
def test_missing_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    cfg = config.load_config(validate_required=True)
    assert cfg["MODEL_NAME"] == "mistral:latest"


@pytest.mark.unit
def test_missing_ollama_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "model")
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    cfg = config.load_config(validate_required=True)
    assert cfg["OLLAMA_API_BASE"] == "http://localhost:11434"


@pytest.mark.unit
def test_empty_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "")
    with pytest.raises(RuntimeError) as exc:
        config.load_config(validate_required=True)
    assert "MODEL_NAME" in str(exc.value)


@pytest.mark.unit
def test_empty_ollama_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "model")
    monkeypatch.setenv("OLLAMA_API_BASE", "")
    with pytest.raises(RuntimeError) as exc:
        config.load_config(validate_required=True)
    assert "OLLAMA_API_BASE" in str(exc.value)


@pytest.mark.unit
def test_missing_redpanda_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "model")
    monkeypatch.delenv("REDPANDA_BROKER", raising=False)
    with pytest.raises(RuntimeError) as exc:
        config.load_config(validate_required=True)
    assert "REDPANDA_BROKER" in str(exc.value)
