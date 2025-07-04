from typing import Any


class Driver:
    def session(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("neo4j driver unavailable")


class GraphDatabase:
    @staticmethod
    def driver(*_a: Any, **_k: Any) -> Driver:
        raise RuntimeError("neo4j driver unavailable")

