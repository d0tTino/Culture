"""Provides a client for interacting with the Ollama LLM service."""

from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, Protocol, TypeVar, cast

from src.shared.typing import (
    JSONDict,
    LLMChatResponse,
    LLMClientMockResponses,
    LLMMessage,
    OllamaGenerateResponse,
    SentimentAnalysisResponse,
    StructuredOutputMock,
)

try:
    import ollama
except ImportError:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning(
        "ollama package not installed; using MagicMock stub for ollama"
    )
    import sys
    from unittest.mock import MagicMock

    ollama = MagicMock()
    sys.modules.setdefault("ollama", ollama)
if TYPE_CHECKING:
    import requests
    from requests.exceptions import RequestException, Timeout
else:
    try:  # pragma: no cover - optional dependency
        import requests
        from requests.exceptions import RequestException, Timeout
    except ImportError:  # pragma: no cover - fallback when requests missing
        logging.getLogger(__name__).warning("requests package not installed; using MagicMock stub")
        from unittest.mock import MagicMock

        requests = MagicMock()

        class RequestException(Exception):
            """Fallback RequestException when requests is unavailable."""

            pass

        class Timeout(RequestException):
            """Fallback Timeout when requests is unavailable."""

            pass


from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from pydantic.v1.fields import ModelField
else:  # pragma: no cover - runtime import
    try:  # Support pydantic >= 2 if installed
        from pydantic.v1.fields import ModelField
    except ImportError:  # pragma: no cover - fallback for pydantic<2
        from pydantic.fields import ModelField  # type: ignore[attr-defined]  # noqa: F401


from src.shared.decorator_utils import monitor_llm_call

from .config import (
    OLLAMA_API_BASE,
    OLLAMA_REQUEST_TIMEOUT,
)
from .ledger import ledger

if TYPE_CHECKING:
    from litellm.exceptions import APIError
else:
    try:
        from litellm.exceptions import APIError
    except ImportError:

        class APIError(Exception):
            """Fallback APIError when litellm is unavailable."""

            pass


_RequestException = RequestException
_APIError = APIError

logger = logging.getLogger(__name__)

# Define generic type variables for Pydantic models and call signatures
T = TypeVar("T")
P = ParamSpec("P")


def charge_du_cost(func: Callable[P, T]) -> Callable[P, T]:
    """Deduct DU cost from the provided agent state."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        state = kwargs.get("agent_state")
        result = func(*args, **kwargs)
        if state is not None:
            try:
                base_price, token_price = ledger.calculate_gas_price(state.agent_id)
                tokens = 1
                if isinstance(result, dict):
                    usage = result.get("usage")
                    if isinstance(usage, dict):
                        tokens = int(usage.get("prompt_tokens", 0)) + int(
                            usage.get("completion_tokens", 0)
                        )
                cost = base_price + token_price * tokens
                if state.du >= cost:
                    state.du -= cost
                    try:
                        ledger.log_change(
                            state.agent_id,
                            0.0,
                            -cost,
                            "llm_gas",
                            gas_price_per_call=base_price,
                            gas_price_per_token=token_price,
                        )
                    except Exception:  # pragma: no cover - optional
                        logger.debug("Ledger logging failed", exc_info=True)
                else:
                    logger.warning(
                        "Insufficient DU for agent %s: cost=%s, available=%s",
                        state.agent_id,
                        cost,
                        state.du,
                    )
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Failed to deduct DU cost: {e}")
        return result

    return wrapper


class OllamaClientProtocol(Protocol):
    """Minimal protocol for the Ollama client used in this module."""

    def chat(
        self: OllamaClientProtocol,
        model: str,
        messages: list[LLMMessage],
        options: dict[str, Any] | None = None,
    ) -> LLMChatResponse: ...


class LLMClientConfig(BaseModel):
    """Simple configuration for ``LLMClient``."""

    model_name: str = "mistral:latest"
    api_key: str | None = None


class LLMClient:
    """Lightweight wrapper around the Ollama client."""

    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config
        self._client = get_ollama_client()

    @monitor_llm_call(model_param="model", context="ollama_chat")
    def chat(
        self,
        model: str,
        messages: list[LLMMessage],
        options: dict[str, Any] | None = None,
    ) -> LLMChatResponse:
        if not self._client:
            raise RuntimeError("Ollama client not initialized")
        return self._client.chat(model=model, messages=messages, options=options)


# Mock implementation variables and functions
_MOCK_ENABLED = False
_MOCK_RESPONSES: LLMClientMockResponses = {
    "default": "This is a mock response from the LLM client.",
    "text_generation": "This is a mock text generation response.",
    "structured_output": {
        "action_intent": "continue_collaboration",
        "reasoning": "Mock reasoning",
        "action": "Mock action",
    },
    "memory_summarization": "This is a mock memory summary.",
    "sentiment_analysis": "positive",
}


def enable_mock_mode(
    enabled: bool = True,
    mock_responses: LLMClientMockResponses | None = None,
) -> None:
    """
    Enable or disable mock mode for testing.

    Args:
        enabled (bool): Whether to enable mock mode
        mock_responses (LLMClientMockResponses | None): Custom mock responses to use
    """
    global _MOCK_ENABLED, _MOCK_RESPONSES
    _MOCK_ENABLED = enabled
    if mock_responses is not None:
        _MOCK_RESPONSES.update(mock_responses)
    logger.info(f"LLM client mock mode {'enabled' if enabled else 'disabled'}")


def is_mock_mode_enabled() -> bool:
    """Return whether mock mode is enabled."""
    return _MOCK_ENABLED


def is_ollama_available() -> bool:
    """
    Check if Ollama service is available.

    Returns:
        bool: True if Ollama is available, False otherwise
    """
    if _MOCK_ENABLED:
        return True  # When in mock mode, pretend Ollama is available

    try:
        # Try to connect to Ollama with a small timeout
        response = requests.get(f"{OLLAMA_API_BASE}", timeout=1)

        return bool(getattr(response, "status_code", 0) == 200)
    except RequestException as e:
        logger.debug(f"Ollama is not available: {e}")
        return False


# Check if OLLAMA_API_BASE is set, if not use the default
if not OLLAMA_API_BASE:
    OLLAMA_API_BASE = "http://localhost:11434"
    logger.warning(f"OLLAMA_API_BASE not set in config, using default: {OLLAMA_API_BASE}")
else:
    logger.info(f"Using OLLAMA_API_BASE: {OLLAMA_API_BASE}")

# Initialize the Ollama client globally using the configured base URL
# This assumes the Ollama service is running and accessible.
client: OllamaClientProtocol | None
try:
    # Ensure OLLAMA_API_BASE is correctly formatted (e.g., 'http://localhost:11434')
    client = cast(OllamaClientProtocol, ollama.Client(host=OLLAMA_API_BASE))
    # Optional: Perform a quick check to see if the client can connect.
    # client.list() # This might be too slow or throw errors if Ollama is busy/starting.
    # A basic check might be better handled during the first actual call.
    logger.info(f"Ollama client initialized for host: {OLLAMA_API_BASE}")
except (APIError, RequestException) as e:
    logger.error(
        f"Failed to initialize Ollama client for host {OLLAMA_API_BASE}: {e}", exc_info=True
    )
    # Set client to None or raise an error if Ollama connection is critical at startup
    client = None
    # Consider raising an error if connection is essential:
    # raise ConnectionError(f"Could not connect to Ollama at {OLLAMA_API_BASE}") from e


def get_ollama_client() -> OllamaClientProtocol | None:
    """Return the initialized Ollama client instance if available."""
    if client is None:
        logger.error("Ollama client is not available. Check connection and configuration.")
    return client


def _retry_with_backoff(
    func: Callable[P, T],
    max_retries: int = 3,
    base_delay: int = 1,
    *args: Any,
    **kwargs: Any,
) -> tuple[T | None, Exception | None]:
    """
    Helper for retrying a function with exponential backoff.
    Returns (result, error) tuple. If successful, error is None.
    """
    e: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs), None
        except (_RequestException, _APIError, ValidationError) as exc:
            e = exc
            logger.error(
                f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}", exc_info=True
            )
            time.sleep(base_delay * (2**attempt))
    return None, e


@charge_du_cost
@monitor_llm_call(model_param="model", context="text_generation")
def generate_text(
    prompt: str,
    model: str = "mistral:latest",
    temperature: float = 0.7,
    *,
    agent_state: Any | None = None,
) -> str | None:
    """
    Generates text using the configured Ollama client.

    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The Ollama model to use (e.g., "mistral:latest", "llama2:latest").
            Ensure this model is pulled in your Ollama instance
            (`ollama pull model_name`).
        temperature (float): The generation temperature (creativity).

    Returns:
        str | None: The generated text, or None if an error occurred or client is unavailable.
    """
    # In mock mode, return a mock response
    if _MOCK_ENABLED:
        logger.debug("Using mock response for text generation")
        # Check for context-specific responses first
        if "summarize" in prompt.lower():
            val = _MOCK_RESPONSES.get("summarization", _MOCK_RESPONSES.get("default"))
            return str(val) if isinstance(val, str) else None
        val = _MOCK_RESPONSES.get("text_generation", _MOCK_RESPONSES.get("default"))
        return str(val) if isinstance(val, str) else None

    ollama_client = get_ollama_client()
    if not ollama_client:
        logger.warning("Attempted to generate text but Ollama client is unavailable.")
        return None

    def call() -> LLMChatResponse:
        messages: list[LLMMessage] = [{"role": "user", "content": prompt}]
        return ollama_client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )

    response, error = _retry_with_backoff(call)

    if error:
        logger.error(f"Failed to generate text after retries: {error}")
        return None
    if isinstance(response, dict) and "message" in response and "content" in response["message"]:
        generated_text = response["message"]["content"]
        logger.debug(f"Received response from Ollama: {generated_text}")
        return str(generated_text).strip()
    else:
        logger.error(f"Unexpected response structure from Ollama: {response}")
        return None


@charge_du_cost
@monitor_llm_call(model_param="model", context="memory_summarization")
def summarize_memory_context(
    memories: list[str],
    goal: str,
    current_context: str,
    model: str = "mistral:latest",
    temperature: float = 0.3,
    *,
    agent_state: Any | None = None,
) -> str:
    """
    Summarizes a list of retrieved memories based on the agent's goal and current context.

    Args:
        memories (List[str]): List of raw memory strings retrieved from the vector store
        goal (str): The agent's goal or objective
        current_context (str): Current context (e.g., previous thought or retrieval query)
        model (str): The Ollama model to use for summarization
        temperature (float): Temperature to control creativity/determinism (lower for
            summarization)

    Returns:
        str: A concise summary of the memories relevant to the goal and context
    """
    if not memories:
        return "(No relevant past memories found via RAG)"

    # In mock mode, return a mock summary
    if _MOCK_ENABLED:
        logger.debug("Using mock response for memory summarization")
        val = _MOCK_RESPONSES.get(
            "memory_summarization",
            f"This is a mock summary of {len(memories)} memories related to '{goal}'.",
        )
        return (
            str(val)
            if isinstance(val, str)
            else f"This is a mock summary of {len(memories)} memories related to '{goal}'."
        )

    ollama_client = get_ollama_client()
    if not ollama_client:
        logger.warning("Attempted to summarize memories but Ollama client is unavailable.")
        return "(Memory summarization failed: LLM client unavailable)"

    # Format memories as a bulleted list for the prompt
    formatted_memories = "\n".join([f"• {memory}" for memory in memories])

    # Construct the summarization prompt
    prompt = (
        f"Summarize the key points from the following memories relevant to the agent's goal "
        f"('{goal}') and the current context ('{current_context}').\n"
        "Be concise and focus on information useful for the agent's next step.\n"
        "Respond ONLY with the summary text.\n\n"
        f"MEMORIES:\n{formatted_memories}\n\nCONCISE SUMMARY:"
    )

    try:
        logger.debug(
            f"Sending memory summarization prompt with {len(memories)} memories, "
            f"goal='{goal}', context='{current_context}'"
        )

        chat_messages: list[LLMMessage] = [{"role": "user", "content": prompt}]
        response = ollama_client.chat(
            model=model,
            messages=chat_messages,
            options={"temperature": temperature},
        )

        # Extract the summary text from the response
        if (
            isinstance(response, dict)
            and "message" in response
            and "content" in response["message"]
        ):
            summary = str(response["message"]["content"]).strip()
            logger.debug(f"Memory summarization result: {summary}")

            # If the summary is empty or too short, return a default message
            if not summary or len(summary) < 10:
                return "(Memory summarization yielded no significant points)"

            return summary
        else:
            logger.error(
                f"Unexpected response structure from Ollama during summarization: {response}"
            )
            return "(Memory summarization failed: Unexpected response format)"

    except (_RequestException, _APIError, ValidationError) as e:
        logger.error(f"Error during memory summarization: {e}", exc_info=True)
        return "(Memory summarization failed due to an error)"


@charge_du_cost
@monitor_llm_call(model_param="model", context="sentiment_analysis")
def analyze_sentiment(
    text: str,
    model: str = "mistral:latest",
    *,
    agent_state: Any | None = None,
) -> float | None:
    """
    Analyzes the sentiment of a given text using Ollama.

    Args:
        text (str): The text to analyze.
        model (str): The Ollama model to use.

    Returns:
        float | None: The sentiment score (0.0 to 1.0) or None if analysis fails.
    """
    # In mock mode, return a mock sentiment
    if _MOCK_ENABLED:
        logger.debug("Using mock response for sentiment analysis")
        val = _MOCK_RESPONSES.get("sentiment_analysis", "neutral")
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                logger.warning(f"Invalid mock sentiment value: {val}")
                return 0.0
        return float(val) if isinstance(val, (int, float)) else 0.0

    ollama_client = get_ollama_client()
    if not ollama_client or not text:
        return None  # Client not available or empty text

    # Simple prompt for sentiment classification
    prompt = (
        f"Analyze the sentiment of the following message. Respond with only one word: "
        f"'positive', 'negative', or 'neutral'.\n\nMessage: \"{text}\"\n\nSentiment:"
    )
    messages: list[LLMMessage] = [{"role": "user", "content": prompt}]
    logger.debug(f"LLM_CLIENT_ANALYZE_SENTIMENT --- Constructed prompt: '''{prompt}'''")

    def call() -> LLMChatResponse:
        return ollama_client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1},  # Low temperature for classification
        )

    response, error = _retry_with_backoff(call)
    if error:
        logger.error(f"Failed to analyze sentiment after retries: {error}")
        return None
    if isinstance(response, dict) and "message" in response and "content" in response["message"]:
        response_content_str = str(response["message"]["content"])
        logger.debug(f"Sentiment analysis: received content string: '{response_content_str}'")
        try:
            sentiment_data: SentimentAnalysisResponse = json.loads(response_content_str)
            if isinstance(sentiment_data, dict) and "sentiment_score" in sentiment_data:
                score = float(sentiment_data["sentiment_score"])
                logger.debug(f"Sentiment analysis result: score '{score}' for text: \"{text}\"")
                return score
            else:
                logger.warning(
                    "Sentiment analysis JSON response missing 'sentiment_score': "
                    f"'{response_content_str}'. Defaulting to 0.0."
                )
                return 0.0  # Default float score
        except json.JSONDecodeError:
            logger.warning(
                "Sentiment analysis failed to parse JSON from response: "
                f"'{response_content_str}'. "
                "Attempting direct string interpretation or defaulting to 0.0."
            )
            # Fallback for direct string if previous mock version sent that.
            # This part may need removal if mocks are consistently JSON.
            sentiment_label_direct = response_content_str.strip().lower()
            if sentiment_label_direct == "positive":
                return 1.0
            if sentiment_label_direct == "negative":
                return -1.0
            if sentiment_label_direct == "neutral":
                return 0.0
            logger.warning(
                f"Could not interpret '{sentiment_label_direct}' as sentiment. Defaulting to 0.0"
            )
            return 0.0  # Default float score
    else:
        logger.error(
            f"Unexpected response structure from Ollama during sentiment analysis: {response}"
        )
        return None  # Or 0.0 if float is always expected


@charge_du_cost
@monitor_llm_call(model_param="model", context="structured_output")
def generate_structured_output(
    prompt: str,
    response_model: type[BaseModel],
    model: str = "mistral:latest",
    temperature: float = 0.2,
    timeout: int | None = None,
    *,
    agent_state: Any | None = None,
) -> BaseModel | None:
    """
    Generate a structured output using the LLM and parse it into the given Pydantic model.
    If mock mode is enabled, returns a mock response that fits the response_model.

    Args:
        prompt (str): Instruction prompt for the LLM
        response_model (Type[T]): The Pydantic model to parse the response into
        model (str): The Ollama model to use
        temperature (float): The temperature for generation
        timeout (int | None): Request timeout in seconds. Defaults to the
            `OLLAMA_REQUEST_TIMEOUT` config value.

    Returns:
        T | None: An instance of the response_model, or None if parsing failed
    """
    # In mock mode, generate a compatible mock response
    if _MOCK_ENABLED:
        logger.debug(f"Using mock response for {response_model.__name__}")
        try:
            model_name = response_model.__name__
            # Ensure response_model is a subclass of BaseModel for type safety
            if not issubclass(response_model, BaseModel):
                raise TypeError("response_model must be a subclass of BaseModel")
            if model_name in _MOCK_RESPONSES:
                mock_data = cast(
                    StructuredOutputMock | str | None, _MOCK_RESPONSES.get(model_name)
                )
                if isinstance(mock_data, dict):
                    mocked_fields: JSONDict = {}
                    mock_fields = getattr(response_model, "model_fields", None)
                    if mock_fields is None:
                        base_fields = getattr(response_model, "__fields__", None)
                        if callable(base_fields):
                            base_fields = base_fields()
                        mock_fields = base_fields or {}
                    for field_name, field in mock_fields.items():
                        if hasattr(field, "is_required") and callable(field.is_required):
                            required = bool(field.is_required())
                        else:
                            required = bool(getattr(field, "required", False))
                        if required:
                            if field.annotation is str:
                                mocked_fields[field_name] = str(
                                    mock_data.get(field_name, f"Mock {field_name}")
                                )
                            elif field.annotation is int:
                                val = mock_data.get(field_name, 1)
                                mocked_fields[field_name] = int(val) if isinstance(val, int) else 1
                            elif field.annotation is float:
                                val = mock_data.get(field_name, 1.0)
                                mocked_fields[field_name] = (
                                    float(val) if isinstance(val, float) else 1.0
                                )
                            elif field.annotation is bool:
                                val = mock_data.get(field_name, False)
                                mocked_fields[field_name] = (
                                    bool(val) if isinstance(val, bool) else False
                                )
                            elif field.annotation is list:
                                val = mock_data.get(field_name, [])
                                mocked_fields[field_name] = val if isinstance(val, list) else []
                            elif field.annotation is dict:
                                val = mock_data.get(field_name, {})
                                mocked_fields[field_name] = val if isinstance(val, dict) else {}
                    return response_model(**mocked_fields)
                else:
                    try:
                        mock_dict = json.loads(str(mock_data))
                        return response_model(**mock_dict)
                    except json.JSONDecodeError:
                        logger.warning("Invalid mock structured output: %s", mock_data)
                        # Fall back to field defaults when mock data is malformed
            # Only define field_defaults if not already defined
            field_defaults: JSONDict = {}
            if hasattr(response_model, "model_fields"):
                fields: Iterable[tuple[str, FieldInfo]] = response_model.model_fields.items()
            else:
                base_fields = getattr(response_model, "__fields__", None)
                if callable(base_fields):
                    base_fields = base_fields()
                fields = base_fields.items() if base_fields is not None else []

            def is_required(f: FieldInfo | Any) -> bool:
                if isinstance(f, FieldInfo):
                    if hasattr(f, "is_required") and callable(f.is_required):
                        return bool(f.is_required())
                    return bool(getattr(f, "required", False))
                return bool(getattr(f, "required", False))

            for field_name, field in fields:
                if is_required(field):
                    if field.annotation is str:
                        field_defaults[field_name] = f"Mock {field_name}"
                    elif field.annotation is int:
                        field_defaults[field_name] = 1
                    elif field.annotation is float:
                        field_defaults[field_name] = 1.0
                    elif field.annotation is bool:
                        field_defaults[field_name] = False
                    elif field.annotation is list:
                        field_defaults[field_name] = []
                    elif field.annotation is dict:
                        field_defaults[field_name] = {}
            return response_model(**field_defaults)
        except (ValidationError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error generating mock structured output: {e}")
            return None
    # Get the Ollama client instance
    ollama_client = get_ollama_client()
    if not ollama_client:
        logger.warning("Attempted to generate structured output but Ollama client is unavailable.")
        return None
    # Ensure response_model is a subclass of BaseModel for type safety
    if not issubclass(response_model, BaseModel):
        raise TypeError("response_model must be a subclass of BaseModel")
    if hasattr(response_model, "model_json_schema"):
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
    else:
        schema_json = json.dumps(response_model.schema(), indent=2)
    example: JSONDict = {}
    example_fields = getattr(response_model, "model_fields", None)
    if example_fields is None:
        base_fields = getattr(response_model, "__fields__", None)
        if callable(base_fields):
            base_fields = base_fields()
        example_fields = base_fields or {}
    for field_name, field in example_fields.items():
        if field.annotation is str:
            example[field_name] = "Example text for " + field_name
        elif field.annotation is str or field.annotation is None:
            example[field_name] = "Optional example for " + field_name
    example_json = json.dumps(example, indent=2)
    structured_prompt = (
        f"{prompt}\n\n"
        "Please respond ONLY with a valid JSON object containing your actual output, "
        "NOT the schema itself.\n"
        f"Schema for reference:\n"
        f"```json\n{schema_json}\n```\n\n"
        f"Example response format (use your own content, not these placeholders):\n"
        f"```json\n{example_json}\n```\n\n"
        f"YOUR RESPONSE:"
    )
    timeout_value = timeout if timeout is not None else OLLAMA_REQUEST_TIMEOUT
    try:
        logger.debug(f"Sending structured prompt to Ollama model '{model}':")
        logger.debug(f"---PROMPT START---\n{structured_prompt}\n---PROMPT END---")
        url = f"{OLLAMA_API_BASE.rstrip('/')}/api/generate"
        response = requests.post(
            url,
            json={
                "model": model,
                "prompt": structured_prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": temperature, "top_p": 0.95, "num_predict": 400},
            },
            timeout=timeout_value,
        )
        response.raise_for_status()
        result: OllamaGenerateResponse = response.json()
        response_text: str = result.get("response", "")
        logger.debug(f"FULL RAW LLM RESPONSE: {response_text}")
        try:
            logger.debug(f"Received potential JSON response from Ollama: {response_text}")
            if response_model:
                json_data: JSONDict = json.loads(str(response_text))
                parsed_output: BaseModel = response_model(**json_data)
                logger.debug(f"Successfully parsed structured output: {parsed_output}")
                return parsed_output
            else:
                # Defensive: fallback for non-model response, cast to BaseModel | None
                return cast(BaseModel | None, json.loads(str(response_text)))
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Failed to parse JSON from Ollama response: {e}")
            logger.warning(f"Raw response: {response_text}")
            return None
    except (RequestException, APIError) as e:
        logger.error(f"Error in generate_structured_output: {e}")

        return None


def get_default_llm_client() -> OllamaClientProtocol | None:
    """
    Creates and returns a default LLM client instance for use in simulations.
    This function is a convenience wrapper that returns the global client.

    Returns:
        The initialized Ollama client instance
    """
    return get_ollama_client()


@charge_du_cost
def generate_response(
    prompt: str,
    model: str = "mistral:latest",
    temperature: float = 0.7,
    *,
    agent_state: Any | None = None,
) -> str | None:
    """
    Generates a response to the given prompt.
    This is an alias for generate_text for backward compatibility.

    If mock mode is enabled, returns a predefined mock response.
    """
    # In mock mode, return predefined response
    if _MOCK_ENABLED:
        logger.debug("Using mock response in generate_response")
        val = _MOCK_RESPONSES.get("default")
        return str(val) if isinstance(val, str) else None

    # Otherwise use the real client
    return generate_text(prompt, model, temperature, agent_state=agent_state)
