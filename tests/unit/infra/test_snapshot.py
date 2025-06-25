import json
from pathlib import Path

import pytest
import zstandard as zstd

from src.infra.snapshot import compute_trace_hash, load_snapshot, save_snapshot

pytestmark = pytest.mark.unit


def test_save_snapshot_compressed(tmp_path: Path) -> None:
    data = {"a": 1}
    save_snapshot(1, data, directory=tmp_path, compress=True)
    fname = tmp_path / "snapshot_1.json.zst"
    assert fname.exists()
    with fname.open("rb") as f:
        decompressed = zstd.ZstdDecompressor().decompress(f.read())
    assert json.loads(decompressed.decode("utf-8")) == data


def test_save_snapshot_compress_without_zstd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {"a": 1}
    from src import infra

    monkeypatch.setattr(infra.snapshot, "zstd", None)

    with pytest.raises(RuntimeError):
        save_snapshot(1, data, directory=tmp_path, compress=True)


def test_load_snapshot_roundtrip(tmp_path: Path) -> None:
    snap = {"step": 1}
    snap["trace_hash"] = compute_trace_hash(snap)
    save_snapshot(1, snap, directory=tmp_path)
    loaded = load_snapshot(1, directory=tmp_path)
    assert loaded == snap


def test_load_snapshot_hash_mismatch(tmp_path: Path) -> None:
    snap = {"step": 1}
    snap["trace_hash"] = compute_trace_hash(snap)
    save_snapshot(1, snap, directory=tmp_path)

    fname = tmp_path / "snapshot_1.json"
    with fname.open() as f:
        data = json.load(f)
    data["step"] = 2
    with fname.open("w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError):
        load_snapshot(1, directory=tmp_path)
