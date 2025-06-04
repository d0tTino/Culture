"""
Integration module for using Ollama models with DSPy.
Provides a proper implementation of DSPy's LM interface for Ollama models.
"""

# mypy: ignore-errors

import logging
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import requests  # type: ignore
from typing_extensions import Self

# Import DSPy and Ollama, providing fallbacks when unavailable
try:
    import dspy

    BaseLM = dspy.LM
    DSPY_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("DSPy not available; using stub implementations")

    class BaseLM:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> list[str]:
            return ["DSPy unavailable"]

    class Signature:
        pass

    class InputField:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class OutputField:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    dspy = SimpleNamespace(
        settings=SimpleNamespace(configure=lambda **_: None),
        LM=BaseLM,
        Signature=Signature,
        InputField=InputField,
        OutputField=OutputField,
    )
    sys.modules.setdefault("dspy", dspy)
    DSPY_AVAILABLE = False

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("ollama package not installed; using MagicMock stub")
    ollama = MagicMock()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dspy_ollama")

__all__ = [
    "OllamaLM",
    "configure_dspy_with_ollama",
    "dspy",
]


class OllamaLM(BaseLM):  # type: ignore[misc]
    """
    A DSPy-compatible language model implementation for Ollama.

    This class implements DSPy's LM interface for local Ollama models,
    ensuring compatibility with DSPy optimizers like BootstrapFewShot.
    """

    def __init__(
        self,
        model_name: str = "mistral:latest",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        request_timeout: None = None,
        **kwargs: object,
    ):
        """
        Initialize the OllamaLM with model configuration.

        Args:
            model_name: Name of the Ollama model to use
            api_base: Base URL for the Ollama API
            temperature: Sampling temperature for generation
            max_tokens: Maximum number of tokens to generate
            request_timeout: Timeout for API requests (in seconds)
            **kwargs: Additional arguments to pass to the Ollama client
        """
        # Initialize the parent class with the model name when DSPy is available
        if DSPY_AVAILABLE:
            super().__init__(model=model_name)
        else:  # pragma: no cover - fallback when DSPy absent
            super().__init__()

        # Store configuration
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.kwargs = kwargs

        # Initialize Ollama client
        try:
            self.client = ollama.Client(host=api_base)
            logger.info(f"Initialized OllamaLM with model {model_name} at {api_base}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise

        # Track statistics for debugging
        self.total_calls = 0
        self.total_tokens = 0
        self.failed_calls = 0

    def basic_request(self: Self, prompt: str, **kwargs: object) -> dict[str, Any]:
        """
        Make a basic request to the Ollama API.
        This is the core method called by DSPy internals.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters to override defaults

        Returns:
            Dict containing the model's response and metadata
        """
        self.total_calls += 1
        start_time = time.time()

        # Merge default kwargs with provided kwargs
        request_kwargs: dict[str, float | int | object] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        request_kwargs.update(self.kwargs)
        request_kwargs.update(kwargs)

        try:
            logger.debug(
                f"Sending request to Ollama: model={self.model_name}, prompt length={len(prompt)}"
            )
            logger.debug(f"Request params: {request_kwargs}")

            # Make the API call to Ollama
            response = self.client.generate(
                model=self.model_name, prompt=prompt, options=request_kwargs
            )
            logger.debug(f"OLLAMA_RAW_RESPONSE ({self.model_name}): {response}")

            # Extract the response text and metadata
            response_text = response.get("response", "")

            # Track token usage if available
            if "eval_count" in response:
                self.total_tokens += response["eval_count"]

            # Format the response for DSPy
            dspy_response = {
                "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
                "model": self.model_name,
                "usage": {
                    "completion_tokens": len(response_text.split()),  # Approximation
                    "prompt_tokens": len(prompt.split()),  # Approximation
                    "total_tokens": len(prompt.split())
                    + len(response_text.split()),  # Approximation
                },
                "raw_ollama_response": response,
            }

            duration = time.time() - start_time
            logger.debug(f"Ollama request completed in {duration:.2f}s")
            return dspy_response

        except Exception as e:
            self.failed_calls += 1
            logger.error(f"Error during Ollama API call: {e}")
            # Return a minimal error response that DSPy can handle
            return {
                "choices": [{"text": f"ERROR: {e!s}", "index": 0, "finish_reason": "error"}],
                "model": self.model_name,
                "error": str(e),
            }

    def __call__(
        self: Self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ) -> list[str]:
        """
        Call the language model with a prompt.
        This is the primary interface used by DSPy modules.

        Args:
            prompt: String prompt (when provided directly)
            messages: List of message dicts for chat models (alternative input method)
            **kwargs: Additional parameters for the language model

        Returns:
            List of completion strings
        """
        # Make sure we have either prompt or messages
        if prompt is None and messages is None:
            logger.error("Either 'prompt' or 'messages' must be provided")
            return ["Error: No prompt or messages provided"]

        # If we have messages but no prompt, convert messages to a prompt string
        if prompt is None and messages is not None:
            logger.debug(f"Converting chat-style messages with {len(messages)} items to string")
            prompt = self._convert_messages_to_string(messages)

        # Make the request
        response = self.basic_request(str(prompt), **kwargs)

        # Extract the completion text
        if response.get("choices"):
            completion = response["choices"][0]["text"]
            completions = [completion]

            # Cache the completions if a cache is configured
            if self.cache and hasattr(self.cache, "store"):
                self.cache.store(prompt, completions)

            return completions
        else:
            logger.warning("Empty or invalid response from Ollama")
            return [""]  # Return empty string as fallback

    def _convert_messages_to_string(self: Self, messages: list[dict[str, Any]]) -> str:
        """
        Convert a list of chat messages to a single string prompt.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            String representation of the chat messages
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")

        return "\n\n".join(prompt_parts)

    def get_stats(self: Self) -> dict[str, Any]:
        """Return usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "failed_calls": self.failed_calls,
        }

    # Implement required methods from the dspy.LM interface
    def generate(self: Self, prompt: str | list[str], **kwargs: object) -> list[str]:
        """
        Generate a completion for the given prompt(s).

        Args:
            prompt: A string or list of strings to generate completions for
            kwargs: Additional keyword arguments for generation

        Returns:
            List of completions for each prompt
        """
        if isinstance(prompt, list):
            # Handle batch of prompts
            results = []
            for p in prompt:
                result = self.__call__(prompt=p)
                results.extend(result)
            return results
        else:
            # Handle single prompt
            return self.__call__(prompt=prompt)

    def _try_load_compiled_program(self: Self, program_path: str) -> bool:
        # Implementation of _try_load_compiled_program method
        # This method should return a boolean indicating whether the program was loaded successfully
        return False  # Placeholder return, actual implementation needed


def configure_dspy_with_ollama(
    model_name: str = "mistral:latest",
    api_base: str = "http://localhost:11434",
    temperature: float = 0.1,
) -> OllamaLM | None:
    """
    Configure DSPy to use an Ollama model globally.

    Args:
        model_name: Name of the Ollama model to use
        api_base: Base URL for the Ollama API
        temperature: Sampling temperature for generation

    Returns:
        The configured OllamaLM instance or None if configuration failed
    """
    if not DSPY_AVAILABLE:
        logger.error("DSPy or Ollama not available. Cannot configure DSPy.")
        return None

    # Check Ollama server availability
    try:
        logger.info(f"Checking Ollama server availability at {api_base}...")
        response = requests.get(api_base, timeout=2)
        if response.status_code != 200:
            logger.error(
                f"Ollama server not accessible at {api_base}. "
                f"Status code: {response.status_code}. "
                "DSPy LM cannot be configured."
            )
            return None
    except Exception as e:
        logger.critical(
            f"Ollama server not accessible at {api_base}. Exception: {e}. "
            "DSPy LM cannot be configured."
        )
        return None

    try:
        # Create and configure the OllamaLM instance
        ollama_lm = OllamaLM(model_name=model_name, api_base=api_base, temperature=temperature)
        dspy.settings.configure(lm=ollama_lm)
        logger.info(
            f"DSPy LM successfully configured with Ollama model '{model_name}' at '{api_base}'."
        )
        return ollama_lm
    except Exception as e:
        logger.error(f"Failed to configure DSPy with Ollama: {e}")
        return None


# === AsyncDSPyManager Design (Task 137 Phase 2) ===


class AsyncDSPyManager:
    """
    Manages asynchronous execution of DSPy program calls with robust error handling and timeouts.
    Intended to allow non-blocking DSPy calls from agents, with result retrieval and
    fallback support.
    """

    def __init__(self: Self, max_workers: int = 4, default_timeout: float = 10.0) -> None:
        """
        Initialize the async manager with a worker pool and default timeout.
        Args:
            max_workers (int): Maximum number of concurrent DSPy calls.
            default_timeout (float): Default timeout for DSPy calls in seconds.
        """

    async def submit(
        self: Self,
        dspy_callable: object,
        *args: object,
        timeout: float | None = None,
        **kwargs: object,
    ) -> object:
        """
        Submit a DSPy program call to be executed asynchronously.
        Args:
            dspy_callable: The DSPy program or function to call.
            *args, **kwargs: Arguments to pass to the DSPy call.
            timeout (float, optional): Timeout for this call.
        Returns:
            An asyncio.Future or similar handle for result retrieval.
        """

    async def get_result(self: Self, future: object, default: object = None) -> object:
        """
        Await and retrieve the result of a submitted DSPy call.
        Args:
            future: The handle returned by submit().
            default: Value to return if the call fails or times out.
        Returns:
            The result of the DSPy call, or the default value on error/timeout.
        """

    async def shutdown(self: Self) -> None:
        """
        Cleanly shut down the async manager, cancelling any pending tasks.
        """


# Usage Example (not implemented):
# manager = AsyncDSPyManager()
# future = await manager.submit(dspy_program, arg1, arg2)
# result = await manager.get_result(future, default="Failsafe output")
