import pytest

from src.agents.memory.multi_layer_retriever import MultiLayerRetriever

pytestmark = pytest.mark.unit


class MockSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MockTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        span = MockSpan(name)
        self.spans.append(span)
        return span


@pytest.mark.asyncio
async def test_retrieve_tracing(monkeypatch):
    tracer = MockTracer()
    monkeypatch.setattr("src.agents.memory.multi_layer_retriever.tracer", tracer)

    class DummyVectorStore:
        async def aretrieve_relevant_memories(self, agent_id, query, k):
            return [{"relevance_score": 1.0}]

    class DummySemantic:
        def retrieve_context_with_scores(self, agent_id, query, k):
            return [{"relevance_score": 2.0}]

    retriever = MultiLayerRetriever(DummyVectorStore(), DummySemantic())
    result = await retriever.retrieve("a", "q", 2)
    assert len(result) == 2

    span_names = [s.name for s in tracer.spans]
    assert "memory.retrieve" in span_names
    assert "memory.episodic_retrieve" in span_names
    assert "memory.semantic_retrieve" in span_names
    for span in tracer.spans:
        assert "memory.latency_ms" in span.attributes
        if span.name != "memory.retrieve":
            assert span.attributes["llm.tokens.prompt"] == 0
            assert span.attributes["llm.tokens.completion"] == 0


@pytest.mark.asyncio
async def test_retrieve_records_latency_metric(monkeypatch):
    times = iter([1.0, 2.0, 3.0, 4.0])

    def fake_perf_counter():
        return next(times)

    monkeypatch.setattr(
        "src.agents.memory.multi_layer_retriever.time.perf_counter", fake_perf_counter
    )

    recorded = {}

    def fake_record_latency(value):
        recorded["latency"] = value

    monkeypatch.setattr(
        "src.infra.metrics.record_retrieval_latency", fake_record_latency
    )

    class DummyVectorStore:
        async def aretrieve_relevant_memories(self, agent_id, query, k):
            return [{"relevance_score": 0.5}]

    retriever = MultiLayerRetriever(DummyVectorStore(), None)
    result = await retriever.retrieve("agent", "query", 1)

    assert result
    assert recorded["latency"] == pytest.approx(3000.0)
