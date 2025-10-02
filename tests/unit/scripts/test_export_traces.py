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


@pytest.mark.unit
def test_export_traces_filter_agent(tmp_path: Path) -> None:
    snap1 = {"step": 1, "agent_id": "a"}
    snap1["trace_hash"] = compute_trace_hash(snap1)
    save_snapshot(1, snap1, directory=tmp_path)
    snap2 = {"step": 2, "agent_id": "b"}
    snap2["trace_hash"] = compute_trace_hash(snap2)
    save_snapshot(2, snap2, directory=tmp_path)
    snap3 = {"step": 3, "agent_id": "a"}
    snap3["trace_hash"] = compute_trace_hash(snap3)
    save_snapshot(3, snap3, directory=tmp_path)

    out = tmp_path / "out.jsonl"
    ret = et.main(
        [
            "--snapshots",
            str(tmp_path),
            "-o",
            str(out),
            "--agent",
            "a",
        ]
    )
    assert ret == 0

    lines = out.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [snap1, snap3]


@pytest.mark.unit
def test_export_latest(tmp_path: Path) -> None:
    snap1 = {"step": 1}
    snap1["trace_hash"] = compute_trace_hash(snap1)
    save_snapshot(1, snap1, directory=tmp_path)
    snap2 = {"step": 2}
    snap2["trace_hash"] = compute_trace_hash(snap2)
    save_snapshot(2, snap2, directory=tmp_path)

    out = tmp_path / "out.jsonl"
    et.export_latest(directory=tmp_path, output=out)

    lines = out.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [snap1, snap2]


@pytest.mark.unit
def test_export_latest_mixed_suffixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "snapshot_1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "snapshot_2.json.zst").write_text("", encoding="utf-8")
    (tmp_path / "snapshot_10.json").write_text("{}", encoding="utf-8")
    (tmp_path / "snapshot_3.json.zst").write_text("", encoding="utf-8")

    calls: list[tuple[int, bool]] = []

    def fake_load_snapshot(step: int, *, directory: Path | str, compress: bool) -> dict[str, int]:
        calls.append((step, compress))
        return {"step": step}

    monkeypatch.setattr(et, "load_snapshot", fake_load_snapshot)

    out = tmp_path / "out.jsonl"
    et.export_latest(directory=tmp_path, output=out)

    lines = out.read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2, 3, 10]
    assert calls == [(1, False), (2, True), (3, True), (10, False)]


@pytest.mark.unit
def test_export_traces_no_snapshots(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    ret = et.main(["--snapshots", str(tmp_path), "-o", str(out)])
    assert ret == 0
    assert out.exists()
    assert out.read_text() == ""


@pytest.mark.unit
def test_export_latest_no_snapshots(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    with pytest.raises(FileNotFoundError):
        et.export_latest(directory=tmp_path, output=out)
