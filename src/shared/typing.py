from typing import Optional, TypedDict


class LLMMessage(TypedDict):
    """Minimal shape for a chat message from Ollama."""

    content: str


class LLMChatResponse(TypedDict):
    """Return type for Ollama chat calls."""

    message: LLMMessage


class OllamaGenerateResponse(TypedDict, total=False):
    """Shape returned from Ollama generate calls."""

    response: str
    done: bool
    eval_count: int
    total_duration: int


class SentimentAnalysisResponse(TypedDict, total=False):
    """Expected JSON structure from the sentiment analysis LLM call."""

    sentiment_score: float
    sentiment_label: str


class StructuredOutputMock(TypedDict, total=False):
    """Mock response structure for structured output generation."""

    action_intent: str
    reasoning: str
    action: str


class LLMClientMockResponses(TypedDict, total=False):
    """Container for predefined mock responses used by the LLM client."""

    default: str
    text_generation: str
    structured_output: StructuredOutputMock
    memory_summarization: str
    sentiment_analysis: str


class SimulationMessage(TypedDict):
    """Message exchanged between agents in the simulation."""

    step: int
    sender_id: str
    recipient_id: Optional[str]
    content: str
    action_intent: Optional[str]
    sentiment_score: Optional[float]
