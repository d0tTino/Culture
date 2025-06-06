import time
from unittest.mock import MagicMock

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
