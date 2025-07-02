from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from src.shared.typing import LLMChatResponse, LLMMessage

from ..config import OLLAMA_API_BASE
from .base import BaseLLMClient

try:  # pragma: no cover - optional dependency
    import ollama
except Exception:  # pragma: no cover - fallback when package missing
    from unittest.mock import MagicMock

    ollama = MagicMock()

logger = logging.getLogger(__name__)


class OllamaLLMClient(BaseLLMClient):
    """Async adapter for the ``ollama`` client."""

    def __init__(self, base_url: str | None = None) -> None:
        host = base_url or OLLAMA_API_BASE
        self._client = ollama.Client(host=host)
        self.base_url = host

    async def chat(self, model: str, messages: list[LLMMessage], **kwargs: Any) -> LLMChatResponse:
        return await asyncio.to_thread(
            self._client.chat,  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            options=kwargs or None,
        )

    async def embed(self, texts: Sequence[str], model: str | None = None) -> list[list[float]]:
        results: list[list[float]] = []
        m = model or "nomic-embed-text"
        for text in texts:
            data = await asyncio.to_thread(
                self._client.embeddings,  # type: ignore[attr-defined]
                prompt=text,
                model=m,
            )
            results.append(data.get("embedding", []))
        return results
