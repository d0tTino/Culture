import uuid
from collections.abc import Generator
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import weaviate.classes as wvc  # For types if needed

from src.agents.memory.weaviate_vector_store_manager import WeaviateVectorStoreManager


@pytest.fixture(scope="function")
def mock_weaviate_client_fixture() -> MagicMock:
    """Mocks the Weaviate client and its relevant methods for schema parsing."""
    mock_client = MagicMock()

    mock_collection_schema_dict = {
        "class": "TestCollection",
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "timestamp", "dataType": ["number"]},
            {"name": "agent_id", "dataType": ["text"]},
            {"name": "uuid", "dataType": ["text"]},
        ],
        "invertedIndexConfig": {
            "bm25": {"b": 0.75, "k1": 1.2},
            "cleanupIntervalSeconds": 60,
            "stopwords": {"preset": "en", "additions": None, "removals": None},
        },
        "shardingConfig": {
            "virtualPerPhysical": 128,
            "desiredCount": 1,
            "actualCount": 1,
            "desiredVirtualCount": 128,
            "actualVirtualCount": 128,
            "key": "_id",
            "strategy": "hash",
            "function": "murmur3",
        },
        "replicationConfig": {"factor": 1},
        "generativeSearch": {},
    }

    mock_collections_api = MagicMock()
    mock_client.collections = mock_collections_api

    def get_side_effect(collection_name_arg: str):
        mock_collection_object = MagicMock(name=collection_name_arg)
        current_schema = mock_collection_schema_dict.copy()
        current_schema["class"] = collection_name_arg
        config_object_mock = MagicMock()
        config_object_mock.get.return_value = current_schema
        mock_collection_object.config = config_object_mock
        mock_collection_object.data = MagicMock()
        mock_collection_object.query = MagicMock()
        return mock_collection_object

    mock_collections_api.exists.return_value = False
    mock_collections_api.get.side_effect = get_side_effect
    mock_collections_api.create.side_effect = (
        lambda name, properties, vectorizer_config: get_side_effect(name)
    )

    return mock_client


@pytest.fixture(scope="function")
def weaviate_manager(
    mock_weaviate_client_fixture: MagicMock,
) -> Generator[WeaviateVectorStoreManager, None, None]:
    test_collection_name = f"TestMemory_{uuid.uuid4().hex[:8]}"
    with patch.object(
        WeaviateVectorStoreManager, "_connect_client", return_value=mock_weaviate_client_fixture
    ):
        manager = WeaviateVectorStoreManager(
            url="http://mocked-url:8080",
            collection_name=test_collection_name,
            embedding_function=lambda x: [float(ord(c)) for c in x][:8],
        )
        manager.client = mock_weaviate_client_fixture
    yield manager
    try:
        pass
    except AssertionError as e:
        print(f"Mock verification failed in teardown: {e}")
    finally:
        if hasattr(manager, "client") and hasattr(manager.client, "close"):
            manager.client.close()


def test_initialization_and_collection_creation(
    weaviate_manager: WeaviateVectorStoreManager, mock_weaviate_client_fixture: MagicMock
) -> None:
    mock_weaviate_client_fixture.collections.exists.return_value = False
    assert weaviate_manager.collection is not None
    mock_weaviate_client_fixture.collections.create.assert_called_once()
    args, kwargs = mock_weaviate_client_fixture.collections.create.call_args
    assert kwargs["name"] == weaviate_manager.collection_name
    assert kwargs["vectorizer_config"] == wvc.config.Configure.Vectorizer.none()


def test_add_and_query_single_memory(
    weaviate_manager: WeaviateVectorStoreManager, mock_weaviate_client_fixture: MagicMock
) -> None:
    text = "hello world"
    meta = {"timestamp": 123.0, "agent_id": "a1", "uuid": str(uuid.uuid4())}
    vector = [1.0] * 8

    mock_object = MagicMock()
    mock_object.properties = {
        "text": text,
        "timestamp": 123.0,
        "agent_id": "a1",
        "uuid": meta["uuid"],
    }
    mock_object.uuid = uuid.UUID(meta["uuid"])
    mock_object.metadata = MagicMock()
    mock_object.metadata.distance = 0.1

    weaviate_manager.collection.query.near_vector.return_value = MagicMock(objects=[mock_object])

    weaviate_manager.add_memories([text], [meta], [vector])

    weaviate_manager.collection.data.insert_many.assert_called_once()
    call_args = weaviate_manager.collection.data.insert_many.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0].properties["text"] == text
    assert call_args[0].vector == vector

    results = weaviate_manager.query_memories(vector, n_results=1)
    assert len(results) == 1
    assert results[0]["text"] == text
    assert results[0]["agent_id"] == "a1"


def test_add_and_query_multiple_memories(
    weaviate_manager: WeaviateVectorStoreManager, mock_weaviate_client_fixture: MagicMock
) -> None:
    texts = [f"text {i}" for i in range(3)]
    metas = [
        {"timestamp": float(i), "agent_id": f"a{i}", "uuid": str(uuid.uuid4())} for i in range(3)
    ]
    vectors = [[float(i)] * 8 for i in range(3)]

    mock_objects = []
    for i in range(3):
        obj = MagicMock()
        obj.properties = {
            "text": texts[i],
            "timestamp": float(i),
            "agent_id": f"a{i}",
            "uuid": metas[i]["uuid"],
        }
        obj.uuid = uuid.UUID(metas[i]["uuid"])
        obj.metadata = MagicMock()
        obj.metadata.distance = 0.1 + i * 0.05
        mock_objects.append(obj)

    weaviate_manager.collection.query.near_vector.return_value = MagicMock(objects=mock_objects)

    weaviate_manager.add_memories(texts, metas, vectors)
    weaviate_manager.collection.data.insert_many.assert_called_once()

    results = weaviate_manager.query_memories(vectors[1], n_results=2)
    assert any(r["text"] == texts[1] for r in results)

    filtered_mock_objects = [mo for mo in mock_objects if mo.properties["agent_id"] == "a1"]
    weaviate_manager.collection.query.near_vector.return_value = MagicMock(
        objects=filtered_mock_objects
    )

    results_filtered = weaviate_manager.query_memories(
        vectors[1], n_results=2, filter_dict={"agent_id": "a1"}
    )
    assert all(r["agent_id"] == "a1" for r in results_filtered)
    last_call_args = weaviate_manager.collection.query.near_vector.call_args
    assert last_call_args is not None
    passed_filters = last_call_args.kwargs.get("filters")
    assert passed_filters is not None
    assert passed_filters.operator.name == "EQUAL"
    assert passed_filters.target == "agent_id"
    assert passed_filters.value == "a1"


def test_query_empty_collection(
    weaviate_manager: WeaviateVectorStoreManager, mock_weaviate_client_fixture: MagicMock
) -> None:
    weaviate_manager.collection.query.near_vector.return_value = MagicMock(objects=[])
    results = weaviate_manager.query_memories([0.0] * 8, n_results=2)
    assert results == []


def test_delete_memories(
    weaviate_manager: WeaviateVectorStoreManager, mock_weaviate_client_fixture: MagicMock
) -> None:
    texts = ["a", "b"]
    metas = [{"timestamp": 1.0, "agent_id": "x", "uuid": str(uuid.uuid4())} for _ in texts]
    vectors = [[1.0] * 8, [2.0] * 8]

    weaviate_manager.add_memories(texts, metas, vectors)

    obj_b = MagicMock()
    obj_b.properties = {"text": "b", "timestamp": 1.0, "agent_id": "x", "uuid": metas[1]["uuid"]}
    obj_b.uuid = uuid.UUID(metas[1]["uuid"])
    obj_b.metadata = MagicMock()
    obj_b.metadata.distance = 0.2
    weaviate_manager.collection.query.near_vector.return_value = MagicMock(objects=[obj_b])

    string_uuid_to_delete_1 = cast(str, metas[0]["uuid"])
    weaviate_manager.delete_memories([string_uuid_to_delete_1])
    weaviate_manager.collection.data.delete_by_id.assert_called_with(string_uuid_to_delete_1)

    results_after_delete_one = weaviate_manager.query_memories(vectors[0], n_results=2)
    assert all(r["uuid"] != string_uuid_to_delete_1 for r in results_after_delete_one)
    if results_after_delete_one:
        assert results_after_delete_one[0]["uuid"] == metas[1]["uuid"]

    weaviate_manager.collection.query.near_vector.return_value = MagicMock(objects=[])
    string_uuid_to_delete_2 = cast(str, metas[1]["uuid"])
    weaviate_manager.delete_memories([string_uuid_to_delete_2])
    weaviate_manager.collection.data.delete_by_id.assert_called_with(string_uuid_to_delete_2)

    results_after_delete_all = weaviate_manager.query_memories(vectors[1], n_results=2)
    assert results_after_delete_all == []


def test_delete_collection(
    weaviate_manager: WeaviateVectorStoreManager, mock_weaviate_client_fixture: MagicMock
) -> None:
    weaviate_manager.delete_collection()
    mock_weaviate_client_fixture.collections.delete.assert_called_with(
        weaviate_manager.collection_name
    )

    mock_weaviate_client_fixture.collections.exists.return_value = False
    assert not mock_weaviate_client_fixture.collections.exists(weaviate_manager.collection_name)
