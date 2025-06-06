from unittest.mock import MagicMock

import pytest

from src.shared.memory_store import WeaviateMemoryStore


@pytest.mark.integration
def test_weaviate_roundtrip() -> None:
    mock_collection = MagicMock()
    mock_collection.data.insert_many = MagicMock()

    result_obj = MagicMock()
    result_obj.properties = {"text": "hello", "author": "A", "timestamp": 0}
    result_obj.uuid = "1"
    result_obj.metadata = MagicMock(distance=0.1)

    mock_collection.query.near_text.return_value = MagicMock(objects=[result_obj])

    mock_collections = MagicMock()
    mock_collections.get.return_value = mock_collection
    mock_collections.exists.return_value = True
    mock_client = MagicMock(collections=mock_collections)

    store = WeaviateMemoryStore(client=mock_client, collection_name="Test")
    store.add_documents(["hello"], [{"author": "A", "timestamp": 0, "uuid": "1"}])
    mock_collection.data.insert_many.assert_called_once()

    results = store.query("hello", top_k=1)
    mock_collection.query.near_text.assert_called_once_with(query="hello", limit=1)
    assert results[0]["content"] == "hello"
    assert results[0]["metadata"]["author"] == "A"
