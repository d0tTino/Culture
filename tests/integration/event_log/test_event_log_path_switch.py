import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.infra import event_log


@pytest.mark.integration
def test_switching_event_log_path_writes_new_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    monkeypatch.setenv("EVENT_LOG_PATH", str(first))
    monkeypatch.setattr(event_log, "_get_producer", lambda: None)
    monkeypatch.setattr(event_log, "_producer", None, raising=False)
    monkeypatch.setattr(event_log, "_last_hash", None, raising=False)
    monkeypatch.setattr(event_log, "_seed", None, raising=False)
    monkeypatch.setattr(event_log, "_seed_cache", {}, raising=False)
    monkeypatch.setattr(event_log, "_header_written", set(), raising=False)
    monkeypatch.setitem(
        sys.modules,
        "src.infra.checkpoint",
        SimpleNamespace(capture_rng_state=lambda: {}),
    )
    event_log.set_seed(4321)

    event_log.log_event({"type": "first", "step": 1})

    monkeypatch.setenv("EVENT_LOG_PATH", str(second))

    logged = event_log.log_event({"type": "second", "step": 2})

    contents = second.read_text(encoding="utf-8").splitlines()
    assert contents, "new event log should not be empty"
    header = json.loads(contents[0])
    assert header["type"] == "header"
    assert header.get("seed") == logged["seed"]
