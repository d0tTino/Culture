import builtins
import importlib
import sys
from collections.abc import Sequence

import pytest


@pytest.mark.unit
@pytest.mark.vector_store
def test_vector_store_import_without_chromadb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing vector_store should succeed when chromadb is missing."""

    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "chromadb" or name.startswith("chromadb."):
            raise ImportError("No module named 'chromadb'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("src.agents.memory.vector_store", None)

    module = importlib.import_module("src.agents.memory.vector_store")
    assert module.chromadb is None
