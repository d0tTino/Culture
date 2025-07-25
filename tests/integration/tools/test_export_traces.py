import json
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
