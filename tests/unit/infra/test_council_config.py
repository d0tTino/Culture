"""Tests for council configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.infra import config as infra_config

pytestmark = pytest.mark.unit


def _reset_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(infra_config, "_COUNCIL_CONFIG", None, raising=False)


def test_load_council_config_returns_defaults_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_cache(monkeypatch)
    missing = tmp_path / "council.yml"
    monkeypatch.setattr(
        infra_config.settings, "COUNCIL_CONFIG_PATH", str(missing), raising=False
    )

    result = infra_config.load_council_config(reload=True)

    assert result["members"], "Expected default members when YAML file is missing"
    assert result["members"][0]["name"] == "Innovator"


def test_load_council_config_reads_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _reset_cache(monkeypatch)
    path = tmp_path / "council.yml"
    roster = {
        "members": [
            {"name": "Strategist", "role": "Planner", "model": "local/mistral"},
            {"name": "Scout", "role": "Observer"},
        ]
    }
    path.write_text(yaml.safe_dump(roster))
    monkeypatch.setattr(infra_config.settings, "COUNCIL_CONFIG_PATH", str(path), raising=False)

    result = infra_config.load_council_config(reload=True)

    assert result["members"][0]["name"] == "Strategist"
    assert result["members"][0]["model"] == "local/mistral"
    assert result["members"][1]["model"], "Missing model should default to base model"
