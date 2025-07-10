import asyncio
import inspect
from collections.abc import Generator
from typing import cast

import pytest
from _pytest.config import Config


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "asyncio: mark test to run with asyncio")


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create and set a new event loop for each test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    marker = pyfuncitem.get_closest_marker("asyncio")
    if marker is not None and inspect.iscoroutinefunction(pyfuncitem.obj):
        loop_obj = pyfuncitem.funcargs.get("event_loop")
        if loop_obj is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            close_loop = True
        else:
            loop = cast(asyncio.AbstractEventLoop, loop_obj)
            close_loop = False
        try:
            args = [pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames]
            loop.run_until_complete(pyfuncitem.obj(*args))
        finally:
            if close_loop:
                loop.close()
                asyncio.set_event_loop(None)
        return True
    return None
