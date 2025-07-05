"""Provide test-friendly stubs for optional dependencies."""

import os
import sys
import types

try:  # pragma: no cover - runtime dependency may be missing
    import neo4j
except Exception:  # pragma: no cover - fallback stub
    neo4j_stub = types.ModuleType("neo4j")

    class GraphDatabase:
        @staticmethod
        def driver(*_a: object, **_k: object) -> None:
            raise RuntimeError("neo4j driver unavailable")

    neo4j_stub.GraphDatabase = GraphDatabase
    neo4j_stub.Driver = object
    sys.modules.setdefault("neo4j", neo4j_stub)
    sys.modules.setdefault("neo4j.exceptions", types.ModuleType("neo4j.exceptions"))
else:  # pragma: no cover - ensure driver attribute exists
    if not hasattr(neo4j.GraphDatabase, "driver"):
        class _FallbackDriver:
            @staticmethod
            def driver(*_a: object, **_k: object) -> None:
                raise RuntimeError("neo4j driver unavailable")

        neo4j.GraphDatabase = _FallbackDriver

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stubs"))

try:  # pragma: no cover - handle langgraph API changes
    import langgraph.graph as _lg_graph
    sys.modules.setdefault("langgraph.graph.graph", _lg_graph)
except Exception:
    pass
