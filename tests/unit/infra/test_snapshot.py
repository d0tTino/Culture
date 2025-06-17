import json
from pathlib import Path

import pytest
import zstandard as zstd

from src.infra.snapshot import save_snapshot

pytestmark = pytest.mark.unit


def test_save_snapshot_compressed(tmp_path: Path) -> None:
    data = {"a": 1}
    save_snapshot(1, data, directory=tmp_path, compress=True)
    fname = tmp_path / "snapshot_1.json.zst"
    assert fname.exists()
    with fname.open("rb") as f:
        decompressed = zstd.ZstdDecompressor().decompress(f.read())
    assert json.loads(decompressed.decode("utf-8")) == data
