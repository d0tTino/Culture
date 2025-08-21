import json
import zipfile
from pathlib import Path

import pytest

from scripts import export_traces
from src.infra import snapshot as snap

pytestmark = pytest.mark.integration


def test_export_traces_sample_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def load_no_verify(
        step: int | str | Path, directory: str | Path = "snapshots", compress: bool | None = None
    ) -> dict:
        path = Path(directory) / f"snapshot_{step}.json"
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    monkeypatch.setattr(snap, "load_snapshot", load_no_verify)
    monkeypatch.setattr(export_traces, "load_snapshot", load_no_verify)

    out = tmp_path / "traces.jsonl"
    ret = export_traces.main(["--snapshots", "snapshots", "-o", str(out)])
    assert ret == 0
    lines = out.read_text().splitlines()
    assert len(lines) > 0
    first = json.loads(lines[0])
    assert isinstance(first, dict)


def test_bundle_run(tmp_path: Path) -> None:
    event_file = tmp_path / "events.jsonl"
    event_file.write_text(json.dumps({"type": "evaluation", "step": 1, "metric": 1}) + "\n")
    snap_dir = tmp_path / "snaps"
    snap_dir.mkdir()
    (snap_dir / "snapshot_0.json").write_text(json.dumps({"step": 0}))
    out = tmp_path / "traces.jsonl"
    bundle = tmp_path / "bundle.zip"
    ret = export_traces.main(
        [
            "--events",
            str(event_file),
            "-o",
            str(out),
            "--snapshots-dir",
            str(snap_dir),
            "--bundle",
            str(bundle),
        ]
    )
    assert ret == 0
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert "traces.jsonl" in names
    assert "metrics.json" in names
    assert "events.jsonl" in names
    assert any(name.startswith("snapshots/") for name in names)
