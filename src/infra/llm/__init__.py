from .base import BaseLLMClient
from .ollama_client import OllamaLLMClient
from .vllm_client import VLLMClient


def get_default_llm_client() -> BaseLLMClient:
    """Return the default LLM client used in simulations."""
    return OllamaLLMClient()


__all__ = ["BaseLLMClient", "OllamaLLMClient", "VLLMClient", "get_default_llm_client"]
