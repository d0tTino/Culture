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
def test_empty_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "")
    with pytest.raises(RuntimeError) as exc:
        config.load_config(validate_required=True)
    assert "MODEL_NAME" in str(exc.value)
