import pathlib
import time
from unittest.mock import MagicMock, patch

import pytest

from src.shared.memory_store import ChromaMemoryStore, MemoryStore, WeaviateMemoryStore


@pytest.mark.unit
def test_chroma_ttl_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    store: MemoryStore = ChromaMemoryStore()
    store.add_documents(["a", "b"], [{"timestamp": 0}, {"timestamp": 10}])
    monkeypatch.setattr(time, "time", lambda: 12)
    store.prune(5)
    results = store.query("", top_k=10)
    assert len(results) == 1
    assert results[0]["content"] == "b"


@pytest.mark.unit
def test_weaviate_ttl_prune(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_collection = MagicMock()
    mock_collection.data.insert_many = MagicMock()
    mock_collection.data.delete_by_id = MagicMock()
    mock_collection.query.near_text = MagicMock(return_value=MagicMock(objects=[]))

    mock_collections = MagicMock()
    mock_collections.get.return_value = mock_collection
    mock_collections.exists.return_value = True
    mock_client = MagicMock(collections=mock_collections)

    store: MemoryStore = WeaviateMemoryStore(client=mock_client, collection_name="Test")
    store.add_documents(
        ["a", "b"],
        [
            {"timestamp": 0, "uuid": "1"},
            {"timestamp": 10, "uuid": "2"},
        ],
    )
    monkeypatch.setattr(time, "time", lambda: 12)
    store.prune(5)
    mock_collection.data.delete_by_id.assert_called_once_with("1")


@pytest.mark.unit
def test_chroma_manager_protocol(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chromadb = pytest.importorskip("chromadb")
    if (
        getattr(chromadb, "PersistentClient", None)
        and chromadb.PersistentClient.__name__ == "_DummyClient"
    ):
        pytest.skip("chromadb stubbed")
    from src.agents.memory.vector_store import ChromaVectorStoreManager

    manager: MemoryStore = ChromaVectorStoreManager(
        persist_directory=str(tmp_path),
        embedding_function=lambda texts: [[0.0] * 8 for _ in texts],
    )
    manager.add_documents(["doc"], [{"timestamp": 0}])
    results = manager.query("doc", top_k=1)
    assert results and results[0]["content"] == "doc"
    monkeypatch.setattr(time, "time", lambda: 10)
    manager.prune(5)
    assert manager.query("doc", top_k=1) == []


@pytest.mark.unit
def test_weaviate_manager_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("weaviate")
    pytest.importorskip("weaviate.classes")
    from src.agents.memory.weaviate_vector_store_manager import WeaviateVectorStoreManager

    mock_collection = MagicMock()
    mock_collection.data.insert_many = MagicMock()
    mock_collection.data.delete_by_id = MagicMock()
    mock_collection.query.near_vector = MagicMock(return_value=MagicMock(objects=[]))

    mock_collections = MagicMock()
    mock_collections.get.return_value = mock_collection
    mock_collections.exists.return_value = True
    mock_client = MagicMock(collections=mock_collections)

    with patch.object(WeaviateVectorStoreManager, "_connect_client", return_value=mock_client):
        manager: MemoryStore = WeaviateVectorStoreManager(
            url="http://x", collection_name="Test", embedding_function=lambda x: [0.0]
        )
    manager.add_documents(["doc"], [{"timestamp": 0}])
    monkeypatch.setattr(time, "time", lambda: 10)
    manager.prune(5)
    mock_collection.data.delete_by_id.assert_called_once()


@pytest.mark.unit
def test_chroma_persistence(tmp_path: pathlib.Path) -> None:
    """Data persists when a directory is provided."""
    store = ChromaMemoryStore(persist_directory=str(tmp_path))
    store.add_documents(["persist"], [{"timestamp": 0}])

    reload_store = ChromaMemoryStore(persist_directory=str(tmp_path))
    results = reload_store.query("", top_k=1)
    assert results and results[0]["content"] == "persist"
