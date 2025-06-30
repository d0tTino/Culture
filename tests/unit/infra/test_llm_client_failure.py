# ruff: noqa: E402
from unittest.mock import MagicMock

import pytest

pytest.importorskip("requests")
import requests


@pytest.mark.unit
def test_generate_text_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that generate_text returns None when the underlying client fails."""
    # Arrange
    # Disable the global mocks for this test to isolate the behavior
    monkeypatch.setattr("src.shared.llm_mocks.mock_llm_functions", lambda: None)
    # Patch the chat method on the LLMClient class to simulate a network error
    monkeypatch.setattr(
        "src.infra.llm_client.LLMClient.chat",
        MagicMock(side_effect=requests.exceptions.RequestException("boom")),
    )
    # The generate_text function should catch the exception and return None
    from src.infra.llm_client import generate_text

    # Act
    result = generate_text("any prompt")

    # Assert
    assert result is None
