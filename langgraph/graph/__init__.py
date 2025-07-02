import asyncio
from typing import Any, Callable

END = "END"
START = "START"

class StateGraph:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.nodes: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.edges: dict[str, list[Any]] = {}
        self.entry_point: str | None = None

    def add_node(self, name: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        self.nodes[name] = fn

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def add_edge(self, src: str, dest: str) -> None:
        self.edges.setdefault(src, []).append(dest)

    def add_conditional_edges(
        self,
        src: str,
        condition_fn: Callable[[dict[str, Any]], str],
        mapping: dict[str, str],
    ) -> None:
        self.edges.setdefault(src, []).append((condition_fn, mapping))

    def compile(self) -> Any:
        async def invoke(state: dict[str, Any]) -> dict[str, Any]:
            node = self.entry_point
            while node and node != END:
                fn = self.nodes[node]
                result = fn(state)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, dict):
                    state.update(result)
                next_nodes = self.edges.get(node)
                if not next_nodes:
                    break
                dest = next_nodes[0]
                if isinstance(dest, tuple):
                    cond_fn, mapping = dest
                    key = cond_fn(state)
                    node = mapping.get(key, END)
                else:
                    node = dest
            return state

        class Executor:
            async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
                return await invoke(state)

        return Executor()
