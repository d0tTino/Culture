import pytest

from src.infra import config


@pytest.fixture(autouse=True)
def _configure_council_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    config_path = tmp_path_factory.mktemp("council-config") / "config.yml"
    config_path.write_text("enabled: true\nmembers: []\n")

    monkeypatch.setenv("DEFAULT_LLM_MODEL", "local/default")
    monkeypatch.setenv("COUNCIL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config.settings, "DEFAULT_LLM_MODEL", "local/default", raising=False)
    monkeypatch.setattr(config.settings, "COUNCIL_CONFIG_PATH", str(config_path), raising=False)
    monkeypatch.setattr(
        config,
        "_CONFIG",
        {"DEFAULT_LLM_MODEL": "local/default", "COUNCIL_CONFIG_PATH": str(config_path)},
        raising=False,
    )
    monkeypatch.setattr(config, "_COUNCIL_CONFIG", None, raising=False)

    yield

    monkeypatch.setattr(config, "_COUNCIL_CONFIG", None, raising=False)
