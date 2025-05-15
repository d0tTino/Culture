#!/usr/bin/env python
"""
Provides a MockLLM context manager for LLM-dependent tests.
"""

import logging
from unittest.mock import patch
from contextlib import contextmanager
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@contextmanager
def MockLLM(responses: Optional[Dict[str, Any]] = None, strict_mode: bool = True):
    """
    Context manager for mocking LLM responses in tests.
    
    Args:
        responses (Dict[str, Any], optional): Dictionary of predefined responses for specific prompts.
                                             Use "default" as key for fallback responses.
        strict_mode (bool): When True, raises exception if no matching response is found.
                           Prevents tests from making actual API calls (prevents timeouts).
    
    Example:
        with MockLLM({
            "specific prompt": "specific response", 
            "default": "default response",
            "structured_output": {"thought": "Test thought", "message_content": None}
        }):
            # Run code that uses LLM
    """
    responses = responses or {"default": "Mocked response from MockLLM"}
    
    # Default structured response for agent decision making
    if "structured_output" not in responses:
        responses["structured_output"] = {
            "thought": "Default mocked thought",
            "message_content": None,
            "message_recipient_id": None,
            "action_intent": "continue_collaboration",
            "requested_role_change": None
        }

    def mock_generate_text(*args, **kwargs):
        logger.info("MockLLM: Intercepted generate_text call")
        prompt = kwargs.get("prompt", "") if kwargs else args[0] if args else ""
        return responses.get(prompt, responses.get("default", "Default mock response"))

    def mock_analyze_sentiment(*args, **kwargs):
        logger.info("MockLLM: Intercepted analyze_sentiment call")
        return 0.0  # Neutral sentiment
    
    def mock_summarize_memory_context(*args, **kwargs):
        logger.info("MockLLM: Intercepted summarize_memory_context call")
        return "Mocked memory context summary"
    
    def mock_generate_structured_output(*args, **kwargs):
        logger.info("MockLLM: Intercepted generate_structured_output call")
        if strict_mode and "structured_output" not in responses:
            raise ValueError("No structured_output defined in MockLLM and strict_mode is enabled")
        return responses.get("structured_output")
    
    def mock_generate_response(*args, **kwargs):
        logger.info("MockLLM: Intercepted generate_response call")
        prompt = kwargs.get("prompt", "") if kwargs else args[0] if args else ""
        return responses.get(prompt, responses.get("default", "Default mock response"))
    
    def mock_ollama_chat(*args, **kwargs):
        logger.info("MockLLM: Intercepted Ollama chat call")
        return {"message": {"content": responses.get("default", "Default Ollama response")}}
    
    def mock_dspy_predict(*args, **kwargs):
        logger.info("MockLLM: Intercepted DSPy predict call")
        return {"output": responses.get("dspy_output", "Default DSPy response")}
    
    # Apply all mocks
    with patch('src.infra.llm_client.generate_text', side_effect=mock_generate_text), \
         patch('src.infra.llm_client.analyze_sentiment', side_effect=mock_analyze_sentiment), \
         patch('src.infra.llm_client.summarize_memory_context', side_effect=mock_summarize_memory_context), \
         patch('src.infra.llm_client.generate_structured_output', side_effect=mock_generate_structured_output), \
         patch('src.infra.llm_client.generate_response', side_effect=mock_generate_response), \
         patch('src.infra.llm_client.client.chat', side_effect=mock_ollama_chat), \
         patch('src.infra.llm_client.ollama.chat', side_effect=mock_ollama_chat):
        try:
            # Try to patch DSPy if it's available
            with patch('dspy.Predict.__call__', side_effect=mock_dspy_predict):
                yield
        except ImportError:
            # If DSPy isn't available, proceed without patching it
            yield 