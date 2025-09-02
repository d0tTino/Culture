import sys
from types import SimpleNamespace

import pytest

sys.modules.setdefault("src.governance.law_board", SimpleNamespace(law_board=None))
sys.modules.setdefault("src.governance.service", SimpleNamespace(governance=None))
sys.modules.setdefault("src.infra.ledger", SimpleNamespace(ledger=None))
sys.modules.setdefault("src.infra.snapshot", SimpleNamespace(load_snapshot=lambda *a, **k: None))
sys.modules.setdefault("src.interfaces.metrics", SimpleNamespace())

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
