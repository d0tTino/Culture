#!/usr/bin/env python
"""
Provides mock helpers for LLM-dependent tests.
"""

import logging
import socket
from unittest.mock import MagicMock

import ollama  # Added import

# import dspy # type: ignore[import-untyped] # Removed as OllamaLocal is used directly
from dspy.predict.ollama import OllamaLocal  # type: ignore[import-untyped]
from pytest import MonkeyPatch
from typing_extensions import Self

logger = logging.getLogger(__name__)


def is_ollama_running() -> bool:
    """Check if Ollama server is running by attempting to connect to localhost:11434"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)  # Short timeout for quick check
        s.connect(("localhost", 11434))
        s.close()
        return True
    except (OSError, socket.timeout):
        return False


class MockLLMResponse:
    """Mock response for LLM calls"""

    def __init__(self: Self, content: str = "This is a mock response") -> None:
        self.content: str = content
        self.structured_output: dict[str, str] = {
            "thought": "Mock thought for testing",
            "message_content": "Mock message for testing",
            "action_intent": "idle",
        }
        self.message: dict[str, str] = {"content": content}


def create_mock_ollama_client() -> object:
    """Create a mock Ollama client"""
    mock_client = MagicMock()

    # Mock the chat method
    mock_response = {"message": {"content": "This is a mock Ollama response"}}
    mock_client.chat.return_value = mock_response

    # Mock the generate method
    mock_client.generate.return_value = {"response": "This is a mock Ollama generation"}

    return mock_client


def patch_ollama_functions(monkeypatch: MonkeyPatch) -> None:
    """
    Patch all Ollama-dependent functions for testing

    Args:
        monkeypatch: The pytest monkeypatch fixture
    """
    # Import here to avoid circular imports
    from src.infra import llm_client

    # Create mock responses
    mock_text = "This is a mock generated text"
    mock_sentiment = "neutral"
    mock_summary = "This is a mock memory summary"

    # Patch the main functions
    monkeypatch.setattr(llm_client, "generate_text", lambda *args, **kwargs: mock_text)
    monkeypatch.setattr(llm_client, "analyze_sentiment", lambda *args, **kwargs: mock_sentiment)
    monkeypatch.setattr(
        llm_client, "summarize_memory_context", lambda *args, **kwargs: mock_summary
    )
    monkeypatch.setattr(
        llm_client,
        "generate_structured_output",
        lambda *args, **kwargs: {
            "thought": "Mock thought",
            "message_content": "Mock message",
            "action_intent": "idle",
        },
    )

    # Create a mock client
    mock_client = create_mock_ollama_client()
    monkeypatch.setattr(llm_client, "client", mock_client)
    monkeypatch.setattr(llm_client, "get_ollama_client", lambda: mock_client)

    # Correctly mock the ollama.Client instance and its methods
    mock_ollama_client_instance = MagicMock(spec=ollama.Client)

    # Mock the 'chat' method to return a dictionary with the expected structure
    mock_ollama_client_instance.chat.return_value = {
        "message": {"content": "Mocked Ollama response"}
    }
    # Mock other methods if they are called, e.g., pull, list, etc.
    mock_ollama_client_instance.list.return_value = []  # Example for 'list'

    # Patch the ollama.Client constructor to return our mocked instance
    monkeypatch.setattr("ollama.Client", lambda *args, **kwargs: mock_ollama_client_instance)

    # Mock top-level functions from the ollama package
    monkeypatch.setattr("ollama.list", MagicMock(return_value=[]))
    monkeypatch.setattr(
        "ollama.chat",
        MagicMock(return_value={"message": {"content": "Mocked Ollama chat response"}}),
    )
    monkeypatch.setattr("ollama.show", MagicMock(return_value={}))
    monkeypatch.setattr("ollama.pull", MagicMock())
    monkeypatch.setattr("ollama.generate", MagicMock(return_value=iter([{"response": "mock"}])))

    # Mock dspy.OllamaLocal directly if it's used for dspy.settings.configure(lm=...)
    # This is a more direct approach if dspy.OllamaLocal is instantiated.
    # If dspy.OllamaLocal uses ollama.Client internally, mocking ollama.Client might suffice.
    # For robustness, let's also mock dspy.OllamaLocal if it exists.
    try:
        # Check if dspy and OllamaLocal are available before attempting to patch
        # import dspy
        # from dspy.predict.ollama import OllamaLocal

        # Create a mock instance of OllamaLocal
        mock_dspy_ollama_local_instance = MagicMock(spec=OllamaLocal)
        # Configure its __call__ method or other relevant methods if necessary
        # For example, if it's called like an LLM:
        mock_dspy_ollama_local_instance.return_value = ["Mocked DSPy OllamaLocal response"]

        # Patch dspy.OllamaLocal to return our mock instance
        monkeypatch.setattr(
            "dspy.predict.ollama.OllamaLocal",
            lambda *args, **kwargs: mock_dspy_ollama_local_instance,
        )
        logger.info("Successfully mocked dspy.OllamaLocal.")
    except ImportError:
        logger.warning(
            "DSPy or dspy.OllamaLocal not found, skipping direct mock of dspy.OllamaLocal."
        )
    except AttributeError:
        logger.warning(
            "dspy.OllamaLocal not found (AttributeError), skipping direct mock of dspy.OllamaLocal."
        )

    logger.info("Global Ollama functions and client have been mocked.")
