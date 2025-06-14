import logging

import pytest

from src.agents.memory.vector_store import ChromaVectorStoreManager

pytest.importorskip("chromadb")


@pytest.mark.integration
@pytest.mark.memory
def test_debug_sqlite_env(monkeypatch, chroma_test_dir, caplog):
    monkeypatch.setenv("DEBUG_SQLITE", "1")
    caplog.set_level(logging.DEBUG)
    store = ChromaVectorStoreManager(persist_directory=chroma_test_dir)
    assert store.debug_sqlite is True
    assert store.client is not None
    messages = [rec.message for rec in caplog.records]
    assert any("SQLite debug" in m for m in messages)
