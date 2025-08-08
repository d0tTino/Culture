from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


@contextmanager
def trace_agent_action(action: str, agent_id: str | None = None, **attrs: Any) -> Iterator[None]:
    """Context manager for tracing agent actions.

    Args:
        action: Name of the action to record in the span.
        agent_id: Optional identifier for the agent performing the action.
        **attrs: Additional span attributes.
    """
    span_name = f"agent.{action}"
    with tracer.start_as_current_span(span_name) as span:
        if agent_id is not None:
            span.set_attribute("agent.id", agent_id)
        for key, value in attrs.items():
            span.set_attribute(key, value)
        yield
