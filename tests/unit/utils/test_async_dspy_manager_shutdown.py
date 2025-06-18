import pytest

from src.infra.async_dspy_manager import AsyncDSPyManager


@pytest.mark.unit
def test_shutdown_no_errors() -> None:
    mgr = AsyncDSPyManager(max_workers=1)
    mgr.shutdown()
    assert True
