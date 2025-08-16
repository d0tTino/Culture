from typing import Any

import pytest

from src.interfaces.dashboard_backend import board_payload_to_embed


@pytest.mark.unit
def test_board_payload_to_embed(snapshot: Any) -> None:
    payload = {"agent_id": "agent12345678", "content": "hello", "step": 5}
    embed = board_payload_to_embed(payload)
    assert embed == snapshot
