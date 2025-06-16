"""Simple Redpanda event logging."""

from __future__ import annotations

import json
import os
from typing import Any

try:  # pragma: no cover - optional dependency
    from confluent_kafka import Consumer as KafkaConsumer
    from confluent_kafka import Producer as KafkaProducer
except Exception:  # pragma: no cover - fallback
    KafkaConsumer = KafkaProducer = Any


_broker = os.getenv("REDPANDA_BROKER", "localhost:9092")
_topic = os.getenv("REDPANDA_TOPIC", "culture.events")

_producer: Any | None = None

_consumer_conf = {
    "bootstrap.servers": _broker,
    "group.id": os.getenv("REPLAY_GROUP", "culture-replay"),
    "auto.offset.reset": "earliest",
}


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


def fetch_events(after_step: int = 0) -> list[dict[str, Any]]:
    """Retrieve events from Redpanda after ``after_step``."""
    if os.getenv("ENABLE_REDPANDA", "0") != "1":
        return []
    events: list[dict[str, Any]] = []
    try:
        consumer = KafkaConsumer(_consumer_conf)
        consumer.subscribe([_topic])
        while True:
            msg = consumer.poll(0.1)
            if msg is None:
                break
            if msg.error():
                break
            try:
                event = json.loads(msg.value().decode("utf-8"))
            except Exception:
                continue
            if event.get("step", 0) > after_step:
                events.append(event)
    except Exception as exc:  # pragma: no cover - best effort
        import logging

        logging.getLogger(__name__).debug("Failed to fetch events: %s", exc)
    finally:
        try:
            consumer.close()
        except Exception:  # pragma: no cover - ignore
            pass
    return events
