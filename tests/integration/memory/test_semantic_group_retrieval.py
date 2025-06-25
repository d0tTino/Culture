import unittest

import pytest

pytest.importorskip("chromadb")

from src.agents.memory.semantic_memory_manager import SemanticMemoryManager
from src.agents.memory.vector_store import ChromaVectorStoreManager


@pytest.mark.integration
@pytest.mark.memory
@pytest.mark.usefixtures("chroma_test_dir")
class TestSemanticGroupingRetrieval(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _inject(self, request, chroma_test_dir):
        self.request = request
        self.chroma_test_dir = chroma_test_dir

    def setUp(self):
        self.vector_store = ChromaVectorStoreManager(persist_directory=self.chroma_test_dir)
        self.manager = SemanticMemoryManager(self.vector_store, driver=None)
        self.agent_id = "semantic_agent"
        for i in range(5):
            self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=i,
                event_type="thought",
                content=f"Cat likes napping {i}",
                memory_type="raw",
            )
        for i in range(5):
            self.vector_store.add_memory(
                agent_id=self.agent_id,
                step=10 + i,
                event_type="thought",
                content=f"Dog enjoys walking {i}",
                memory_type="raw",
            )
        self.manager.group_memories_by_topic(self.agent_id, num_topics=2)

    def tearDown(self):
        client = getattr(self.vector_store, "client", None)
        if client and hasattr(client, "close"):
            client.close()

    def test_hit_rate(self):
        cat_res = self.manager.retrieve_context(self.agent_id, "sleepy cat", k=5)
        dog_res = self.manager.retrieve_context(self.agent_id, "walk with dog", k=5)
        cat_hits = sum(1 for m in cat_res if "cat" in m["content"].lower())
        dog_hits = sum(1 for m in dog_res if "dog" in m["content"].lower())
        total_hits = cat_hits + dog_hits
        total = len(cat_res) + len(dog_res)
        hit_rate = total_hits / total if total else 0
        assert hit_rate > 0.7
