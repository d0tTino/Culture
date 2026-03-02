import pytest
from pydantic import ValidationError

from src.interfaces.domain_command_adapters import parse_bus_command
from src.interfaces.interaction_commands import InteractionContext
from src.interfaces.interaction_schema import (
    InjectEventEnvelope,
    KnowledgeBoardEnvelope,
    parse_interaction_envelope,
)

pytestmark = pytest.mark.unit


def test_parse_bus_command_accepts_modern_text_payload() -> None:
    envelope = parse_bus_command(
        {"intent": "knowledge_board", "text": "hello", "sender_id": "u-1", "source": "dashboard"}
    )

    assert isinstance(envelope, KnowledgeBoardEnvelope)
    assert envelope.text == "hello"


def test_parse_bus_command_supports_legacy_fields_with_deprecation_warning() -> None:
    with pytest.deprecated_call(match="deprecated"):
        envelope = parse_bus_command(
            {
                "command_type": "inject_event",
                "content": "storm incoming",
                "prompt": "global",
                "agent_id": "mod-1",
            },
            context=InteractionContext(sender_id="mod-1", source="discord", permissions={"admin"}),
        )

    assert isinstance(envelope, InjectEventEnvelope)
    assert envelope.text == "storm incoming"
    assert envelope.scope == "global"


def test_parse_bus_command_rejects_forbidden_fields_for_intent() -> None:
    with pytest.raises(ValidationError):
        parse_interaction_envelope({"intent": "knowledge_board", "text": "hi", "action": "pause"})
