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
    monkeypatch.setattr(infra_config.settings, "COUNCIL_CONFIG_PATH", str(missing), raising=False)

    result = infra_config.load_council_config(reload=True)

    assert result["members"], "Expected default members when YAML file is missing"
    assert result["members"][0]["display_name"] == "Innovator"
    assert result["members"][0]["member_id"] == "innovator"
    assert result["max_concurrent_calls"] == infra_config.settings.COUNCIL_MAX_CONCURRENT_CALLS
    assert result["du_budget_per_question"] == infra_config.settings.DU_BUDGET_PER_QUESTION
    assert result["voting_mode"] == "single_winner"
    assert result["members"][0]["is_active"] is True
    assert result["members"][0]["temperature"] == pytest.approx(0.4)


def test_load_council_config_reads_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _reset_cache(monkeypatch)
    path = tmp_path / "council.yml"
    roster = {
        "members": [
            {
                "id": "strategist",
                "name": "Strategist",
                "role": "Innovator",
                "model": "local/mistral",
            },
            {"id": "scout", "name": "Scout", "role": "Analyzer"},
        ]
    }
    path.write_text(yaml.safe_dump(roster))
    monkeypatch.setattr(infra_config.settings, "COUNCIL_CONFIG_PATH", str(path), raising=False)

    result = infra_config.load_council_config(reload=True)

    assert result["members"][0]["display_name"] == "Strategist"
    assert result["members"][0]["model"] == "local/mistral"
    assert result["members"][1]["model"], "Missing model should default to base model"
    assert result["members"][1]["persona"], "Missing persona should be derived"
    assert all(member.get("is_active") for member in result["members"])
    assert all("temperature" in member for member in result["members"])


def test_load_council_config_merges_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_cache(monkeypatch)
    path = tmp_path / "council.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "members": [{"id": "strategist", "name": "Strategist", "role": "Innovator"}],
                "max_concurrent_calls": 5,
            }
        )
    )
    monkeypatch.setattr(infra_config.settings, "COUNCIL_CONFIG_PATH", str(path), raising=False)

    result = infra_config.load_council_config(reload=True)

    assert result["max_concurrent_calls"] == 5
    assert result["du_budget_per_question"] == infra_config.settings.DU_BUDGET_PER_QUESTION
    assert result["members"][0]["model"], "Missing model should default to base model"
    assert result["voting_mode"] == "single_winner"
