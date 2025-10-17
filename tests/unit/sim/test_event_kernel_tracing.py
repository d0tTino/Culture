import pytest

from src.sim.event_kernel import EventKernel

pytestmark = pytest.mark.unit


class RecordingSpan:
    def __init__(self, tracer: "RecordingTracer", name: str) -> None:
        self._tracer = tracer
        self.name = name
        self.attributes: dict[str, object] = {}

    def __enter__(self) -> "RecordingSpan":
        self._tracer.spans.append(self)
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    def start_as_current_span(self, name: str) -> RecordingSpan:
        return RecordingSpan(self, name)


@pytest.mark.asyncio
async def test_dispatch_records_span_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = RecordingTracer()
    monkeypatch.setattr("src.sim.event_kernel.tracer", tracer)

    kernel = EventKernel()
    kernel.set_budget("agent-1", 10)
    kernel.set_budget("agent-2", 10)
    called: list[str] = []

    async def cb(label: str) -> None:
        called.append(label)

    kernel.schedule_immediate_nowait(lambda: cb("first"), agent_id="agent-1", tokens=3)
    kernel.schedule_in_nowait(0, lambda: cb("second"), agent_id="agent-2", tokens=1)

    await kernel.dispatch(2)

    assert called == ["first", "second"]
    assert len(tracer.spans) == 2

    first_span, second_span = tracer.spans
    assert first_span.name == "event.kernel.dispatch"
    assert first_span.attributes == {
        "event.agent_id": "agent-1",
        "event.step": 0,
        "event.token_burn": 3,
        "event.queue_depth_before_callback": 1,
    }
    assert second_span.attributes == {
        "event.agent_id": "agent-2",
        "event.step": 0,
        "event.token_burn": 1,
        "event.queue_depth_before_callback": 0,
    }

