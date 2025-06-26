import json
from pathlib import Path

import pytest
import zstandard as zstd

from src import infra
from src.infra.snapshot import (
    compute_trace_hash,
    load_snapshot,
    save_snapshot,
    upload_snapshot,
)

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


@pytest.mark.unit
def test_snapshot_s3_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("boto3")
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    with mock_aws():
        monkeypatch.setattr(infra.snapshot, "boto3", boto3)
        monkeypatch.setattr(infra.snapshot, "_s3_client", None)
        monkeypatch.setattr(infra.snapshot, "S3_BUCKET", "bucket", raising=False)
        monkeypatch.setattr(infra.snapshot, "S3_PREFIX", "prefix", raising=False)

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="bucket")

        snap = {"step": 1}
        snap["trace_hash"] = compute_trace_hash(snap)
        save_snapshot(1, snap, directory=tmp_path)
        upload_snapshot(1, directory=tmp_path)
        (tmp_path / "snapshot_1.json").unlink()

        loaded = load_snapshot(1, directory=tmp_path)
        assert loaded == snap
