from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from src.shared.typing import LLMChatResponse, LLMMessage

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class VLLMClient(BaseLLMClient):
    """Async client for a vLLM server speaking the OpenAI API."""

    def __init__(self, base_url: str = "http://localhost:8001/v1") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def chat(self, model: str, messages: list[LLMMessage], **kwargs: Any) -> LLMChatResponse:
        payload = {"model": model, "messages": messages} | kwargs
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice: dict[str, Any] = data.get("choices", [{}])[0]
        msg: dict[str, Any] = choice.get("message", {})
        return {
            "message": {"role": msg.get("role", "assistant"), "content": msg.get("content", "")}
        }

    async def embed(self, texts: Sequence[str], model: str | None = None) -> list[list[float]]:
        payload: dict[str, Any] = {"input": list(texts)}
        if model:
            payload["model"] = model
        resp = await self._client.post("/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return [item.get("embedding", []) for item in data.get("data", [])]
