import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from src.infra import llm_client


class DummyModel(BaseModel):
    foo: str


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_generate_structured_output_uses_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        url = args[1] if len(args) > 1 else args[0]
        captured["url"] = url
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.text = json.dumps({"response": json.dumps({"foo": "bar"})})
        return resp

    monkeypatch.setattr(llm_client, "USE_VLLM", False)
    monkeypatch.setattr(
        "src.infra.llm_client.httpx.AsyncClient.post",
        AsyncMock(side_effect=fake_post),
    )

    result = await llm_client.async_generate_structured_output("prompt", DummyModel)

    assert isinstance(result, DummyModel)
    assert result.foo == "bar"
    assert captured["url"].endswith("/api/generate")


@pytest.mark.unit
def test_sync_wrapper_calls_async(monkeypatch: pytest.MonkeyPatch) -> None:
    async_called = False

    async def fake_async(*args: object, **kwargs: object) -> DummyModel:
        nonlocal async_called
        async_called = True
        return DummyModel(foo="bar")

    monkeypatch.setattr(llm_client, "async_generate_structured_output", fake_async)

    result = llm_client.generate_structured_output("prompt", DummyModel)

    assert async_called is True
    assert isinstance(result, DummyModel)
    assert result.foo == "bar"
