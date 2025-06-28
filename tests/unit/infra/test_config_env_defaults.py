import pytest

from src.infra import config

pytestmark = pytest.mark.unit


def test_load_config_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "123")
    cfg = config.load_config(validate_required=False)
    assert cfg["OLLAMA_REQUEST_TIMEOUT"] == 123
