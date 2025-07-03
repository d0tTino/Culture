"""Compatibility shim for old langgraph import path."""

END = "__end__"
START = "__start__"


class StateGraph:
    """Placeholder StateGraph for type checking."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def add_node(self, *args: object, **kwargs: object) -> None: ...

    def set_entry_point(self, *args: object, **kwargs: object) -> None: ...

    def add_edge(self, *args: object, **kwargs: object) -> None: ...

    def add_conditional_edges(self, *args: object, **kwargs: object) -> None: ...

    def compile(self) -> object: ...


__all__ = ["END", "START", "StateGraph"]
