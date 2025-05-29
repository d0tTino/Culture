import logging
import time
from collections.abc import Generator

import pytest
from pytest import LogCaptureFixture

from src.utils.async_dspy_manager import AsyncDSPyManager


@pytest.fixture(scope="module")
def manager() -> Generator[AsyncDSPyManager, None, None]:
    mgr = AsyncDSPyManager(max_workers=2, default_timeout=1.0)
    yield mgr
    mgr.shutdown()


@pytest.mark.asyncio
async def test_successful_execution(manager: AsyncDSPyManager, caplog: LogCaptureFixture) -> None:
    def mock_dspy_success(duration: float) -> str:
        time.sleep(duration)
        return "success"

    with caplog.at_level(logging.INFO):
        future = await manager.submit(mock_dspy_success, 0.1)
        result = await manager.get_result(
            future, default_value="fail", dspy_callable=mock_dspy_success
        )
    assert result == "success"
    assert any("completed successfully" in r for r in caplog.text.splitlines())


@pytest.mark.asyncio
async def test_timeout(manager: AsyncDSPyManager, caplog: LogCaptureFixture) -> None:
    def mock_dspy_success(duration: float) -> str:
        time.sleep(duration)
        return "success"

    with caplog.at_level(logging.WARNING):
        future = await manager.submit(mock_dspy_success, 2.0)
        result = await manager.get_result(
            future, default_value="timeout", timeout=0.2, dspy_callable=mock_dspy_success
        )
    assert result == "timeout"
    assert any("Timeout awaiting result" in r for r in caplog.text.splitlines())


@pytest.mark.asyncio
async def test_execution_error(manager: AsyncDSPyManager, caplog: LogCaptureFixture) -> None:
    def mock_dspy_error() -> None:
        raise ValueError("DSPy error")

    with caplog.at_level(logging.ERROR):
        future = await manager.submit(mock_dspy_error)
        result = await manager.get_result(
            future, default_value="error", dspy_callable=mock_dspy_error
        )
    assert result == "error"
    assert any("Exception retrieving result" in r for r in caplog.text.splitlines())


def def_noop_manager() -> AsyncDSPyManager:
    mgr = AsyncDSPyManager(max_workers=2, default_timeout=1.0)
    return mgr
