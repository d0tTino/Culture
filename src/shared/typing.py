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


class SimulationMessage(TypedDict):
    """Message exchanged between agents in the simulation."""

    step: int
    sender_id: str
    recipient_id: Optional[str]
    content: str
    action_intent: Optional[str]
