from __future__ import annotations

import types

import httpx
import pytest

from src.extensions import register_widget_backend


class DummyClient:
    async def __aenter__(self) -> DummyClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        pass

    async def post(self, *args: object, **kwargs: object) -> None:
        raise httpx.HTTPError("boom")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_widget_backend_handles_http_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: DummyClient())
    with caplog.at_level("ERROR"):
        await register_widget_backend("Widget", "http://x/y.js")
    assert "Widget registration failed" in caplog.text
