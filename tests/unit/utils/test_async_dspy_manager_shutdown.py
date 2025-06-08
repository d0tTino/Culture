import pytest

from src.shared.async_utils import AsyncDSPyManager


@pytest.mark.unit
def test_shutdown_no_errors() -> None:
    mgr = AsyncDSPyManager(max_workers=1)
    mgr.shutdown()
    assert True
