import os
from pathlib import Path

import pytest

from src.utils.paths import ensure_dir, get_temp_dir


@pytest.mark.unit
def test_get_temp_dir_creates_directory(tmp_path: Path) -> None:
    temp = get_temp_dir(prefix="culture_test_")
    try:
        assert temp.exists() and temp.is_dir()
    finally:
        # Cleanup
        os.rmdir(temp)


@pytest.mark.unit
def test_ensure_dir(tmp_path: Path) -> None:
    target = tmp_path / "foo" / "bar"
    result = ensure_dir(target)
    assert result == target
    assert target.exists() and target.is_dir()
