from typing import Any

from . import _types

class Client:
    def __init__(self, host: str = ...) -> None: ...
    def generate(self, *, model: str, prompt: str, options: _types.Options) -> dict[str, Any]: ...

__all__ = ["Client", "_types"]
