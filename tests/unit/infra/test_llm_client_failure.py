from unittest.mock import MagicMock

import pytest

pytest.importorskip("requests")
import requests  # noqa: E402
from pytest import MonkeyPatch  # noqa: E402

from src.infra import llm_client  # noqa: E402


@pytest.mark.unit
def test_generate_text_failure(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "get_ollama_client",
        lambda: MagicMock(
            chat=MagicMock(side_effect=requests.exceptions.RequestException("boom"))
        ),
    )
    monkeypatch.setattr(
        llm_client,
        "_retry_with_backoff",
        lambda func, *a, **kw: (None, requests.exceptions.RequestException("boom")),
    )
    result = llm_client.generate_text("hi")
    assert result is None
