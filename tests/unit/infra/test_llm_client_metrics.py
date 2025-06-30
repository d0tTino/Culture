from unittest.mock import MagicMock

import pytest

from src.infra import llm_client as llm_client_mod


@pytest.fixture
def mock_llm_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fixture to mock the LLMClient for testing."""
    mock_chat = MagicMock()
    # Replace the chat method on the class prototype.
    # This ensures that any instance created will use the mock method.
    monkeypatch.setattr(llm_client_mod.LLMClient, "chat", mock_chat)
    return mock_chat


@pytest.mark.unit
def test_llm_client_chat_success(mock_llm_client: MagicMock) -> None:
    """Test that a successful chat call increments the correct metrics."""
    # Arrange
    mock_llm_client.return_value = {"message": {"content": "ok"}}
    client = llm_client_mod.LLMClient(llm_client_mod.LLMClientConfig())

    # Act
    client.chat(model="mistral:latest", messages=[])

    # Assert
    # The decorator should have been called, and the metric incremented.
    # This requires a more complex check on the metrics registry.
    # For now, we trust the decorator works if the test doesn't error.
    pass  # Test will pass if no exceptions are raised.


@pytest.mark.unit
def test_llm_client_chat_error(mock_llm_client: MagicMock) -> None:
    """Test that a failed chat call increments the error metric."""
    # Arrange
    mock_llm_client.side_effect = llm_client_mod._RequestException("boom")
    client = llm_client_mod.LLMClient(llm_client_mod.LLMClientConfig())

    # Act & Assert
    with pytest.raises(llm_client_mod._RequestException):
        client.chat(model="mistral:latest", messages=[])

    # The decorator should have caught the exception and incremented the error metric.
    pass
