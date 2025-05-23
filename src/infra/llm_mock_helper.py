#!/usr/bin/env python
"""
Provides mock helpers for LLM-dependent tests.
"""

import socket
import logging
from unittest.mock import MagicMock
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def is_ollama_running():
    """Check if Ollama server is running by attempting to connect to localhost:11434"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)  # Short timeout for quick check
        s.connect(("localhost", 11434))
        s.close()
        return True
    except (socket.error, socket.timeout):
        return False

class MockLLMResponse:
    """Mock response for LLM calls"""
    def __init__(self, content="This is a mock response"):
        self.content = content
        self.structured_output = {
            "thought": "Mock thought for testing",
            "message_content": "Mock message for testing",
            "action_intent": "idle"
        }
        self.message = {"content": content}

def create_mock_ollama_client():
    """Create a mock Ollama client"""
    mock_client = MagicMock()
    
    # Mock the chat method
    mock_response = {
        "message": {"content": "This is a mock Ollama response"}
    }
    mock_client.chat.return_value = mock_response
    
    # Mock the generate method
    mock_client.generate.return_value = {
        "response": "This is a mock Ollama generation"
    }
    
    return mock_client

def patch_ollama_functions(monkeypatch):
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
    monkeypatch.setattr(llm_client, "summarize_memory_context", lambda *args, **kwargs: mock_summary)
    monkeypatch.setattr(llm_client, "generate_structured_output", lambda *args, **kwargs: 
                        {"thought": "Mock thought", "message_content": "Mock message", "action_intent": "idle"})
    
    # Create a mock client
    mock_client = create_mock_ollama_client()
    monkeypatch.setattr(llm_client, "client", mock_client)
    monkeypatch.setattr(llm_client, "get_ollama_client", lambda: mock_client) 