import json
from pathlib import Path

import pytest

from scripts import export_traces
from src.infra.snapshot import compute_trace_hash, save_snapshot

pytestmark = pytest.mark.unit


def _make_snapshot(step: int, agent_id: str, directory: Path) -> None:
    data = {"step": step, "agent_id": agent_id, "value": step}
    data["trace_hash"] = compute_trace_hash(data)
    save_snapshot(step, data, directory=directory)


def test_export_traces_jsonl(tmp_path: Path) -> None:
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    _make_snapshot(1, "a1", snap_dir)
    _make_snapshot(2, "a1", snap_dir)

    out = tmp_path / "out.jsonl"
    export_traces.main([
        "--snapshots",
        str(snap_dir),
        "--agent",
        "a1",
        "--start-step",
        "2",
        "-o",
        str(out),
    ])

    lines = out.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["step"] == 2
    assert record["agent_id"] == "a1"
