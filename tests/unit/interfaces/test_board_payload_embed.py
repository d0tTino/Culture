import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

try:  # pragma: no cover - optional dependency
    import src.governance.law_board as _law_board  # noqa: F401
except Exception:  # pragma: no cover - fallback stub
    sys.modules.setdefault("src.governance.law_board", SimpleNamespace(law_board=None))

try:  # pragma: no cover - optional dependency
    import src.governance.service as _governance_service  # noqa: F401
except Exception:  # pragma: no cover - fallback stub
    sys.modules.setdefault(
        "src.governance.service",
        SimpleNamespace(
            governance=None,
            GovernanceService=type("GovernanceService", (), {}),
        ),
    )

try:  # pragma: no cover - optional dependency
    import src.infra.ledger as _ledger  # noqa: F401
except Exception:  # pragma: no cover - fallback stub
    sys.modules.setdefault("src.infra.ledger", SimpleNamespace(ledger=None))

try:  # pragma: no cover - optional dependency
    import src.infra.snapshot as _snapshot  # noqa: F401
except Exception:  # pragma: no cover - fallback stub
    sys.modules.setdefault(
        "src.infra.snapshot",
        SimpleNamespace(
            load_snapshot=lambda *a, **k: None,
            save_snapshot=lambda *a, **k: None,
            upload_snapshot=lambda *a, **k: None,
            compute_trace_hash=lambda data: hashlib.sha256(
                json.dumps(data, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        ),
    )

try:  # pragma: no cover - optional dependency
    import src.interfaces.metrics as _metrics  # noqa: F401
except Exception:  # pragma: no cover - fallback stub
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
