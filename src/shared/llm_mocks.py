#!/usr/bin/env python
"""
Provides mock helpers for LLM-dependent tests.
"""

import json
import logging
import socket
from typing import Any
from unittest.mock import MagicMock

# Handle optional Ollama dependency
try:
    import ollama
except ImportError:
    logging.getLogger(__name__).warning(
        "ollama package not installed; using MagicMock stub for ollama"
    )
    ollama = MagicMock()

# Handle optional DSPy OllamaLocal dependency
try:
    from dspy.predict.ollama import OllamaLocal
except ImportError:
    logging.getLogger(__name__).warning(
        "dspy.predict.ollama not available; using stub OllamaLocal"
    )

    class OllamaLocal:  # type: ignore[no-redef]
        """
        Stub for DSPy OllamaLocal when dspy.predict.ollama is unavailable.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> list[str]:
            return [f"Stubbed OllamaLocal response to prompt: {prompt}"]


from pytest import MonkeyPatch
from typing_extensions import Self

logger = logging.getLogger(__name__)

# Module-level mock responses for broader access
mock_text_global = "This is a mock generated text (global)"
mock_summary_global = "This is a mock memory summary (global)"


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


def create_mock_ollama_client() -> MagicMock:
    """Create a mock Ollama client"""
    original_client = getattr(ollama, "Client", MagicMock)
    if isinstance(original_client, MagicMock):  # If already mocked, avoid spec
        mock_client = MagicMock()
    else:
        mock_client = MagicMock(spec=original_client)  # Use spec for better mocking

    # Mock the chat method with more specific behavior for sentiment
    def mock_chat_for_sentiment_and_general(
        *args: Any, **kwargs: Any
    ) -> dict[str, dict[str, str]]:
        messages = kwargs.get("messages", [])
        prompt_content = ""
        if (
            messages
            and isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], dict)
        ):
            prompt_content = messages[0].get("content", "")
        logger.debug(f"MOCK_CHAT_HELPER --- Received prompt_content: '''{prompt_content}'''")

        # Check if it's a sentiment analysis prompt (more specific check)
        if "Analyze the sentiment of the following message." in prompt_content:
            if "Strongly disagree" in prompt_content:
                logger.debug("Mock ollama.Client.chat: returning -0.7 sentiment for disagreement")
                return {
                    "message": {
                        "content": '{"sentiment_score": -0.7, "sentiment_label": "negative"}'
                    }
                }
            elif (
                "vital discussion" in prompt_content
                or "perspectives constructively" in prompt_content
            ):
                logger.debug("Mock ollama.Client.chat: returning 0.2 sentiment for facilitation")
                return {
                    "message": {
                        "content": '{"sentiment_score": 0.2, "sentiment_label": "neutral"}'
                    }
                }
            else:
                logger.debug(
                    f"Mock ollama.Client.chat: returning 0.0 sentiment for: {prompt_content[:30]}..."
                )
                return {
                    "message": {
                        "content": '{"sentiment_score": 0.0, "sentiment_label": "neutral"}'
                    }
                }

        # Fallback for other chat calls (e.g., from DSPy OllamaLM not directly using generate_structured_output)
        # This is the part that might be causing the "expected string or buffer" if DSPy gets this
        # For now, let's assume DSPy structured calls go through the more specific ollama.generate mock later
        logger.debug(
            f"Mock ollama.Client.chat: returning generic mock for other prompt: {prompt_content[:50]}..."
        )
        # Ensure this default response is what other direct .chat() users might expect if not sentiment.
        # The original code had "This is a mock Ollama response". Let's stick to that for non-sentiment.
        return {"message": {"content": "This is a mock Ollama response"}}

    mock_client.chat.side_effect = mock_chat_for_sentiment_and_general

    # Mock the generate method - this is often used by DSPy
    # Keep the existing heuristic for generate, but ensure it returns a dict with "response" key
    # as ollama.Client.generate does.

    def mock_generate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        prompt_content = str(kwargs.get("prompt", ""))  # Ensure prompt_content is a string
        logger.debug(f"MOCK_GENERATE_PROMPT_CONTENT_DEBUG: '''{prompt_content}'''")

        # Determine which DSPy program this prompt is for based on unique field combinations
        is_l1_summary_prompt = (
            "Your output fields are:" in prompt_content
            and "`l1_summary` (str)" in prompt_content
            and "`recent_events` (str)" in prompt_content
            and "`agent_role` (str)" in prompt_content
        )
        is_action_intent_prompt = (
            "Your output fields are:" in prompt_content
            and "`chosen_action_intent` (str)" in prompt_content
            and "`justification_thought` (str)" in prompt_content
            and "`available_actions` (str)" in prompt_content  # Added for specificity
        )
        is_role_thought_prompt = (
            "Your output fields are:" in prompt_content
            and "`thought` (str)" in prompt_content
            and "`role_name` (str)" in prompt_content
            and "`context` (str)" in prompt_content
        )

        if is_l1_summary_prompt:
            logger.debug(
                "Mock ollama.Client.generate: returning JSON structure for L1SummaryGenerator"
            )
            summary_value_str = "Mock L1 Summary from global: " + mock_summary_global
            response_json_str = json.dumps({"l1_summary": summary_value_str})
            return {
                "response": response_json_str,
                "done": True,
                "eval_count": 10,
                "total_duration": 100,
            }

        elif is_action_intent_prompt:
            logger.debug(
                "Mock ollama.Client.generate: returning action intent structure for ActionIntentSelector"
            )
            action_intent_content = {
                "chosen_action_intent": "send_direct_message",  # Mocked action
                "justification_thought": "This is a mock justification for selecting send_direct_message from mock_generate.",
            }
            response_json_str = json.dumps(action_intent_content)
            return {
                "response": response_json_str,
                "done": True,
                "eval_count": 10,
                "total_duration": 100,
            }

        elif is_role_thought_prompt:
            logger.debug(
                "Mock ollama.Client.generate: returning thought structure for RoleThoughtGenerator"
            )
            thought_content = {
                "thought": "As a MockRole, this is a generic mocked thought for RoleThoughtGenerator from mock_generate."
            }
            response_json_str = json.dumps(thought_content)
            return {
                "response": response_json_str,
                "done": True,
                "eval_count": 10,
                "total_duration": 100,
            }

        # Fallback for any other direct ollama.generate calls that don't match DSPy signatures
        # This might also catch DSPy prompts if the above conditions are not met perfectly.
        else:
            logger.warning(
                f"Mock ollama.Client.generate: FALLBACK for unrecognized prompt structure. Prompt content: {prompt_content[:200]}..."
            )
            # Generic JSON response that might or might not work depending on caller
            fallback_content = {
                "detail": "Fallback mock response from generate",
                "prompt_received": prompt_content[:100],
            }
            response_json_str = json.dumps(fallback_content)
            return {
                "response": response_json_str,
                "done": True,
                "eval_count": 5,
                "total_duration": 50,
            }

    mock_client.generate = MagicMock(side_effect=mock_generate)
    # Add other common methods if needed, e.g., pull, list
    mock_client.list.return_value = []

    return mock_client


def patch_ollama_functions(monkeypatch: MonkeyPatch) -> None:
    """
    Patch all Ollama-dependent functions for testing

    Args:
        monkeypatch: The pytest monkeypatch fixture
    """
    # Import here to avoid circular imports
    from src.infra import llm_client

    # Create mock responses (these are now global, but can be referenced here if needed for clarity or local overrides)
    # mock_text = mock_text_global (no, lambdas below will capture the global directly)
    # mock_summary = mock_summary_global

    # Remove the direct patch of llm_client.analyze_sentiment
    # The logic is now handled by the more sophisticated mock_client.chat side_effect

    # Patch the main functions that DON'T rely on llm_client.client directly
    # if they have their own logic or simpler mock needs.
    monkeypatch.setattr(llm_client, "generate_text", lambda *args, **kwargs: mock_text_global)
    monkeypatch.setattr(
        llm_client, "summarize_memory_context", lambda *args, **kwargs: mock_summary_global
    )
    # generate_structured_output in llm_client uses llm_client.client.generate, so it will use the mock_client's generate.

    # Create and set the more intelligent mock client for llm_client.py
    intelligent_mock_client: MagicMock = create_mock_ollama_client()
    monkeypatch.setattr(llm_client, "client", intelligent_mock_client)
    monkeypatch.setattr(llm_client, "get_ollama_client", lambda: intelligent_mock_client)

    # Correctly mock the ollama.Client constructor to return our intelligent_mock_client
    # This ensures any *new* instance of ollama.Client gets this behavior.
    monkeypatch.setattr("ollama.Client", lambda *args, **kwargs: intelligent_mock_client)

    # Mock top-level functions from the ollama package that might be used by DSPy adapters directly
    monkeypatch.setattr("ollama.list", MagicMock(return_value=[]))
    monkeypatch.setattr("ollama.pull", MagicMock())
    monkeypatch.setattr(
        "ollama.show", MagicMock(return_value={})
    )  # Added show as it was in previous version

    # For ollama.chat and ollama.generate at the package level (used by DSPy)
    # they should also use the sophisticated logic from intelligent_mock_client.
    # We can achieve this by making them delegate to the methods of an instance of intelligent_mock_client.
    # Note: This is a bit tricky because intelligent_mock_client is already a MagicMock itself.
    # We want ollama.chat to behave like intelligent_mock_client.chat

    # The most robust way is to ensure ollama.Client always returns our configured mock,
    # and that dspy.OllamaLocal (if used) also gets this configured client or is mocked appropriately.

    # The setattr("ollama.Client", ...) above should cover cases where dspy instantiates its own Ollama client.
    # If dspy calls ollama.chat or ollama.generate directly (as static/module functions),
    # we need to patch those too, to use the same logic.

    # Use the side_effects from the intelligent_mock_client for the package level mocks
    monkeypatch.setattr(
        "ollama.chat",
        MagicMock(side_effect=intelligent_mock_client.chat.side_effect),
    )
    monkeypatch.setattr(
        "ollama.generate",
        MagicMock(side_effect=intelligent_mock_client.generate.side_effect),
    )

    # Mock dspy.OllamaLocal directly if it's used for dspy.settings.configure(lm=...)
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


async def get_mock_sentiment_analysis(
    text: str, agent_id: str, current_step: int
) -> dict[str, Any]:
    """
    Mock sentiment analysis function.
    """
    logger.debug(
        f"Mock sentiment analysis called for agent {agent_id} at step {current_step} with text: '{text[:50]}...'"
    )
    # Simulate some processing if needed, or just return a mock value
    # await asyncio.sleep(0) # Optional: if you want to ensure it's treated as a coroutine
    return {
        "sentiment_score": 0.1,
        "sentiment_label": "slightly_positive",
        "analysis_source": "mock_helper",
    }
