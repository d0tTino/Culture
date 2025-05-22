import uuid

import pytest

from src.agents.memory.weaviate_vector_store_manager import WeaviateVectorStoreManager


@pytest.fixture(scope="function")
def weaviate_manager():
    # Use a unique collection name per test for isolation
    test_collection = f"TestMemory_{uuid.uuid4().hex[:8]}"
    manager = WeaviateVectorStoreManager(
        url="http://localhost:8080",
        collection_name=test_collection,
        embedding_function=lambda x: [float(ord(c)) for c in x][:8],  # Dummy embedding
    )
    yield manager
    # Teardown: delete the collection
    try:
        manager.delete_collection()
        manager.close()
    except Exception:
        pass


def test_initialization_and_collection_creation(
    weaviate_manager: WeaviateVectorStoreManager,
) -> None:
    # Should not raise and collection should exist
    assert weaviate_manager.collection is not None
    # Use v4 API: check existence directly
    assert weaviate_manager.client.collections.exists(weaviate_manager.collection_name)


def test_add_and_query_single_memory(weaviate_manager: WeaviateVectorStoreManager):
    text = "hello world"
    meta = {"timestamp": 123, "agent_id": "a1", "uuid": str(uuid.uuid4())}
    vector = [1.0] * 8
    weaviate_manager.add_memories([text], [meta], [vector])
    # Query
    results = weaviate_manager.query_memories(vector, n_results=1)
    assert len(results) == 1
    assert results[0]["text"] == text
    assert results[0]["agent_id"] == "a1"


def test_add_and_query_multiple_memories(weaviate_manager: WeaviateVectorStoreManager):
    texts = [f"text {i}" for i in range(3)]
    metas = [{"timestamp": i, "agent_id": f"a{i}", "uuid": str(uuid.uuid4())} for i in range(3)]
    vectors = [[float(i)] * 8 for i in range(3)]
    weaviate_manager.add_memories(texts, metas, vectors)
    # Query for one
    results = weaviate_manager.query_memories(vectors[1], n_results=2)
    assert any(r["text"] == texts[1] for r in results)
    # Query with filter
    results = weaviate_manager.query_memories(
        vectors[1], n_results=2, filter_dict={"agent_id": "a1"}
    )
    assert all(r["agent_id"] == "a1" for r in results)


def test_query_empty_collection(weaviate_manager: WeaviateVectorStoreManager):
    results = weaviate_manager.query_memories([0.0] * 8, n_results=2)
    assert results == []


def test_delete_memories(weaviate_manager: WeaviateVectorStoreManager):
    texts = ["a", "b"]
    metas = [{"timestamp": 1, "agent_id": "x", "uuid": str(uuid.uuid4())} for _ in texts]
    vectors = [[1.0] * 8, [2.0] * 8]
    weaviate_manager.add_memories(texts, metas, vectors)
    # Delete one
    weaviate_manager.delete_memories([metas[0]["uuid"]])
    results = weaviate_manager.query_memories(vectors[0], n_results=2)
    assert all(r["uuid"] != metas[0]["uuid"] for r in results)
    # Delete all
    weaviate_manager.delete_memories([metas[1]["uuid"]])
    results = weaviate_manager.query_memories(vectors[1], n_results=2)
    assert results == []


def test_delete_collection(weaviate_manager: WeaviateVectorStoreManager):
    # Should delete without error
    weaviate_manager.delete_collection()
    # Collection should not exist anymore (v4 API)
    assert not weaviate_manager.client.collections.exists(weaviate_manager.collection_name)
