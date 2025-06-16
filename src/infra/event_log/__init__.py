"""Simple Redpanda event logging."""

from __future__ import annotations

import json
import os
from typing import Any

try:  # pragma: no cover - optional dependency
    from confluent_kafka import Producer as KafkaProducer
except Exception:  # pragma: no cover - fallback
    KafkaProducer = Any


_broker = os.getenv("REDPANDA_BROKER", "localhost:9092")
_topic = os.getenv("REDPANDA_TOPIC", "culture.events")

_producer: Any | None = None


def _get_producer() -> Any:
    global _producer
    if _producer is None:
        _producer = KafkaProducer({"bootstrap.servers": _broker})
    return _producer


def log_event(event: dict[str, Any]) -> None:
    """Send an event dictionary to Redpanda."""
    if os.getenv("ENABLE_REDPANDA", "0") != "1":
        return
    try:
        payload = json.dumps(event).encode("utf-8")
        producer = _get_producer()
        producer.produce(_topic, payload)
        producer.poll(0)
    except Exception as exc:  # pragma: no cover - best effort
        import logging

        logging.getLogger(__name__).debug("Failed to log event: %s", exc)
