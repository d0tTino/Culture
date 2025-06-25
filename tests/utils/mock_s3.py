import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any


class DummyS3Client:
    """A minimal in-memory S3 client for tests."""

    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or tempfile.mkdtemp())
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, bucket: str, key: str) -> Path:
        path = self.base_dir / bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def upload_file(self, Filename: str | Path, Bucket: str, Key: str) -> None:
        shutil.copyfile(Filename, self._path(Bucket, Key))

    def download_file(self, Bucket: str, Key: str, Filename: str | Path) -> None:
        shutil.copyfile(self._path(Bucket, Key), Filename)

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:
        with self._path(Bucket, Key).open("wb") as f:
            f.write(Body)

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": self._path(Bucket, Key).open("rb")}


def setup_mock_s3() -> DummyS3Client:
    """Install a lightweight ``boto3`` stub returning :class:`DummyS3Client`."""

    client = DummyS3Client()

    if "boto3" in sys.modules:
        boto3 = sys.modules["boto3"]
    else:
        boto3 = types.ModuleType("boto3")
        sys.modules["boto3"] = boto3

    def client_factory(service_name: str, *args: Any, **kwargs: Any) -> Any:
        if service_name == "s3":
            return client
        raise NotImplementedError(f"Unsupported service {service_name}")

    boto3.client = client_factory  # type: ignore[attr-defined]

    return client
