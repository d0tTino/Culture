from pathlib import Path

import pytest

from src.infra import config, settings


@pytest.mark.unit
def test_env_file_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import reload

    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_API_BASE=http://from_env\n")
    monkeypatch.chdir(tmp_path)

    # Reload the settings and config modules to force them to re-read the .env file
    # in the new current directory.
    reload(settings)
    reload(config)

    # Now, creating a new instance should pick up the value from the .env file.
    test_settings = settings.ConfigSettings()
    assert test_settings.OLLAMA_API_BASE == "http://from_env"

    monkeypatch.setenv("REDPANDA_BROKER", "broker")
    monkeypatch.setenv("OPA_URL", "http://opa")
    cfg = config.load_config(validate_required=True)
    assert cfg["OLLAMA_API_BASE"] == "http://from_env"
