from pathlib import Path

import pytest

pytest.importorskip("zstandard")
mock_s3 = pytest.importorskip("tests.utils.mock_s3")

from src.infra.snapshot import load_snapshot, save_snapshot

setup_mock_s3 = mock_s3.setup_mock_s3

pytestmark = pytest.mark.integration


@pytest.fixture()
def s3_client(tmp_path: Path):
    client = setup_mock_s3()
    # store objects under tmp_path for easy cleanup
    client.base_dir = tmp_path
    return client


def test_upload_and_download_snapshot(tmp_path: Path, s3_client) -> None:
    data = {"msg": "hello"}
    save_snapshot(1, data, directory=tmp_path, compress=True)
    src = tmp_path / "snapshot_1.json.zst"
    assert src.exists()

    s3_client.upload_file(str(src), "bucket", "snap.zst")

    src.unlink()
    assert not src.exists()

    s3_client.download_file("bucket", "snap.zst", src)

    loaded = load_snapshot(src)
    assert loaded == data


def test_put_and_get_snapshot(tmp_path: Path, s3_client) -> None:
    data = {"val": 42}
    save_snapshot(2, data, directory=tmp_path, compress=True)
    src = tmp_path / "snapshot_2.json.zst"

    with src.open("rb") as f:
        s3_client.put_object(Bucket="bucket", Key="obj.zst", Body=f.read())

    src.unlink()

    body = s3_client.get_object(Bucket="bucket", Key="obj.zst")["Body"]
    out = tmp_path / "downloaded.zst"
    with out.open("wb") as f:
        f.write(body.read())

    loaded = load_snapshot(out)
    assert loaded == data
