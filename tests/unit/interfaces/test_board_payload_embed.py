import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.interfaces.dashboard_backend import board_payload_to_embed  # noqa: E402


@pytest.mark.unit
def test_board_payload_to_embed() -> None:
    payload = {"agent_id": "agent12345678", "content": "hello", "step": 5}
    embed = board_payload_to_embed(payload)
    assert embed == {
        "author": {"name": "Posted by Agent agent123"},
        "color": 0xFFD700,
        "description": "```hello```",
        "title": "📝 New Knowledge Board Entry (Step 5)",
    }
