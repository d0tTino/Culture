from typing import Any


class Client:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> dict[str, dict[str, str]]:
        content = messages[-1]["content"] if messages else ""
        return {"message": {"content": content}}

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        return {"response": ""}

    def list(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    stream: bool = False,
) -> dict[str, dict[str, str]]:
    content = messages[-1]["content"] if messages else ""
    return {"message": {"content": content}}


def generate(*args: Any, **kwargs: Any) -> dict[str, str]:
    return {"response": ""}


def list(*args: Any, **kwargs: Any) -> list[Any]:
    return []


def pull(*args: Any, **kwargs: Any) -> None:
    pass


def show(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {}
