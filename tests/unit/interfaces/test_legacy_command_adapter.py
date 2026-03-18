from datetime import date

import pytest

from src.interfaces.legacy_command_adapter import (
    assert_no_expired_legacy_adapter_usage,
    get_legacy_adapter_hits,
    normalize_legacy_command_payload,
    reset_legacy_adapter_hits,
)

pytestmark = pytest.mark.unit


def setup_function() -> None:
    reset_legacy_adapter_hits()


def teardown_function() -> None:
    reset_legacy_adapter_hits()


def test_legacy_adapter_emits_structured_telemetry() -> None:
    payload = normalize_legacy_command_payload(
        {"command_type": "inject_event", "content": "storm", "prompt": "global"},
        adapter="test.adapter",
        context_source="discord",
        sender_id="user-1",
    )

    hits = get_legacy_adapter_hits()
    assert payload["intent"] == "inject_event"
    assert payload["text"] == "storm"
    assert payload["scope"] == "global"
    assert len(hits) == 1
    assert hits[0].asdict()["adapter"] == "test.adapter"
    assert hits[0].asdict()["fields"] == ["command_type", "content", "prompt"]


def test_legacy_adapter_fails_after_migration_window() -> None:
    normalize_legacy_command_payload({"command_type": "human_message", "content": "hello"})

    with pytest.raises(AssertionError, match="Legacy command adapter usage remains"):
        assert_no_expired_legacy_adapter_usage(today=date(2027, 1, 1))
