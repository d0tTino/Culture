
from pathlib import Path

import pytest

from src.infra import config
from src.infra.settings import ConfigSettings


@pytest.mark.unit
def test_env_file_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_API_BASE=http://from_env\n")
    monkeypatch.chdir(tmp_path)
    settings = ConfigSettings()
    assert settings.OLLAMA_API_BASE == "http://from_env"

    monkeypatch.setenv("REDPANDA_BROKER", "broker")
    monkeypatch.setenv("OPA_URL", "http://opa")
    monkeypatch.chdir(tmp_path)
    cfg = config.load_config(validate_required=True)
    assert cfg["OLLAMA_API_BASE"] == "http://from_env"
