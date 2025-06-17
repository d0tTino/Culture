from unittest.mock import MagicMock

import pytest

from src.infra import config
from src.utils.policy import evaluate_with_opa


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluate_with_opa_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": {"allow": False, "content": "filtered"}}
    monkeypatch.setattr("src.utils.policy.requests.post", MagicMock(return_value=mock_resp))
    allowed, new_content = await evaluate_with_opa("test")
    assert allowed is False
    assert new_content == "filtered"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluate_with_opa_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(config._CONFIG, "OPA_URL", "http://opa")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"result": {"allow": True}}
    monkeypatch.setattr("src.utils.policy.requests.post", MagicMock(return_value=mock_resp))
    allowed, new_content = await evaluate_with_opa("hello")
    assert allowed is True
    assert new_content == "hello"
