import pytest

from src.infra.settings import ConfigSettings


@pytest.mark.unit
def test_settings_propagate_env(monkeypatch):
    monkeypatch.setenv("HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("HTTP_PORT", "1234")
    monkeypatch.setenv("ENABLE_REDPANDA", "1")
    monkeypatch.setenv("OPA_BLOCKLIST", "foo,bar")
    settings = ConfigSettings()
    assert settings.HTTP_HOST == "127.0.0.1"
    assert settings.HTTP_PORT == 1234
    assert settings.ENABLE_REDPANDA is True
    assert settings.OPA_BLOCKLIST == "foo,bar"
