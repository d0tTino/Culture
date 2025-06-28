import pytest

from src.infra import config


@pytest.mark.unit
def test_load_config_missing_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDPANDA_BROKER", raising=False)
    monkeypatch.delenv("OPA_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    with pytest.raises(RuntimeError) as exc:
        config.load_config(validate_required=True)
    msg = str(exc.value)
    assert "REDPANDA_BROKER" in msg
    assert "OPA_URL" in msg
    assert "MODEL_NAME" not in msg


@pytest.mark.unit
def test_load_config_with_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDPANDA_BROKER", "localhost:9092")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    cfg = config.load_config(validate_required=True)
    assert cfg["REDPANDA_BROKER"] == "localhost:9092"
    assert cfg["OPA_URL"] == "http://opa"
    assert cfg["MODEL_NAME"] == "test-model"
