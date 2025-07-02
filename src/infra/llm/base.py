from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from src.shared.typing import LLMChatResponse, LLMMessage


class BaseLLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def chat(self, model: str, messages: list[LLMMessage], **kwargs: Any) -> LLMChatResponse:
        """Send a chat completion request."""
        raise NotImplementedError

    @abstractmethod
    async def embed(self, texts: Sequence[str], model: str | None = None) -> list[list[float]]:
        """Return embeddings for the given texts."""
        raise NotImplementedError
