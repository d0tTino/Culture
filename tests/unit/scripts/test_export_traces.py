import json
from pathlib import Path

import pytest

from scripts import export_traces as et
from src.infra.snapshot import compute_trace_hash, save_snapshot


@pytest.mark.unit
def test_export_traces_snapshots(tmp_path: Path) -> None:
    snap1 = {"step": 1}
    snap1["trace_hash"] = compute_trace_hash(snap1)
    save_snapshot(1, snap1, directory=tmp_path)
    snap2 = {"step": 2}
    snap2["trace_hash"] = compute_trace_hash(snap2)
    save_snapshot(2, snap2, directory=tmp_path)

    out = tmp_path / "out.jsonl"
    ret = et.main(["--snapshots", str(tmp_path), "-o", str(out)])
    assert ret == 0

    lines = out.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [snap1, snap2]
