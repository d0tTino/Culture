import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.infra import llm_client


class DummyModel(BaseModel):
    foo: str


@pytest.mark.unit
def test_generate_structured_output_uses_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_post(url: str, *args: object, **kwargs: object) -> MagicMock:
        captured["url"] = url
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"response": json.dumps({"foo": "bar"})}
        return resp

    monkeypatch.setattr(llm_client, "OLLAMA_API_BASE", "http://override:1234")
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    result = llm_client.generate_structured_output("prompt", DummyModel)

    assert isinstance(result, DummyModel)
    assert result.foo == "bar"
    assert captured["url"] == "http://override:1234/api/generate"
