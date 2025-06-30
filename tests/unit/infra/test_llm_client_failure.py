from unittest.mock import MagicMock

import pytest

pytest.importorskip("requests")
import requests


@pytest.mark.unit
@pytest.mark.disable_global_llm_mock
def test_generate_text_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that generate_text returns None when the underlying client fails."""
    # Arrange
    # Patch the mock mode check to ensure we test the real failure path.
    monkeypatch.setattr("src.infra.llm_client.is_mock_mode_enabled", lambda: False)

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
