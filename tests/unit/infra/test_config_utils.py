import importlib

import pytest

from src.infra import config


@pytest.mark.unit
def test_get_relationship_label_ranges() -> None:
    assert config.get_relationship_label(-0.8) == "Hostile"
    assert config.get_relationship_label(0.0) == "Neutral"
    assert config.get_relationship_label(0.6) == "Positive"


@pytest.mark.unit
def test_get_config_value_with_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(config.CONFIG_OVERRIDES, "TEST_KEY", "from_override")
    assert (
        config.get_config_value_with_override("TEST_KEY", default="def", module_name="src.infra.config")
        == "from_override"
    )


@pytest.mark.unit
def test_get_config_value_with_missing_module() -> None:
    assert (
        config.get_config_value_with_override("MISSING", default="def", module_name="nonexistent.module")
        == "def"
    )
