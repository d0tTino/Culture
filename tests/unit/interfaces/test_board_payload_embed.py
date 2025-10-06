import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

class _DummyGovernanceService:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


sys.modules.setdefault("src.governance.law_board", SimpleNamespace(law_board=None))
sys.modules.setdefault(
    "src.governance.service",
    SimpleNamespace(GovernanceService=_DummyGovernanceService, governance=None),
)
sys.modules.setdefault(
    "src.infra.ledger",
    SimpleNamespace(
        ledger=SimpleNamespace(log_penalty=lambda *a, **k: None),
        log_penalty=lambda *a, **k: None,
    ),
)
def _compute_trace_hash(data: dict[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save_snapshot(
    step: int,
    data: dict[str, object],
    *,
    directory: str | Path = "snapshots",
    compress: bool | None = None,
) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"snapshot_{step}.json"
    with file_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


sys.modules.setdefault(
    "src.infra.snapshot",
    SimpleNamespace(
        load_snapshot=lambda *a, **k: None,
        upload_snapshot=lambda *a, **k: None,
        save_snapshot=_save_snapshot,
        compute_trace_hash=_compute_trace_hash,
    ),
)
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
