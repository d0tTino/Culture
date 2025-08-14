"""Simple Redpanda event logging."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    from confluent_kafka import Consumer as KafkaConsumer
    from confluent_kafka import Producer as KafkaProducer
except Exception:  # pragma: no cover - fallback
    KafkaConsumer = KafkaProducer = Any

from src.infra.snapshot import compute_trace_hash

_broker = os.getenv("REDPANDA_BROKER", "localhost:9092")
_topic = os.getenv("REDPANDA_TOPIC", "culture.events")

_producer: Any | None = None
_last_hash: str | None = None
_seed: int | None = None

_consumer_conf = {
    "bootstrap.servers": _broker,
    "group.id": os.getenv("REPLAY_GROUP", "culture-replay"),
    "auto.offset.reset": "earliest",
}


def _log_file(path: str | Path | None = None) -> Path:
    """Return the event log file path."""
    if path is not None:
        return Path(path)
    return Path(os.getenv("EVENT_LOG_PATH", "event_log.jsonl"))


def _is_valid_event(
    event: dict[str, Any], last_step: int, last_hash: str | None
) -> bool:
    """Check ``event`` ordering and integrity."""
    step = event.get("step", 0)
    if step <= last_step:
        return False
    event_copy = {**event}
    trace_hash = event_copy.pop("trace_hash", None)
    if trace_hash != compute_trace_hash(event_copy):
        return False
    if last_hash is not None and event.get("prev_hash") != last_hash:
        return False
    return True


def _filter_events(
    events: Iterable[dict[str, Any]], *, after_step: int = 0
) -> list[dict[str, Any]]:
    """Return events sorted by step and validated against tampering."""
    last_step = after_step
    last_hash: str | None = None
    valid: list[dict[str, Any]] = []
    for ev in events:
        if _is_valid_event(ev, last_step, last_hash):
            last_step = ev.get("step", last_step)
            last_hash = ev.get("trace_hash")
            valid.append(ev)
    return valid


def _get_producer() -> Any:
    global _producer
    if _producer is None:
        _producer = KafkaProducer({"bootstrap.servers": _broker})
    return _producer


def _iter_file_events(
    start_tick: int = 0,
    end_tick: int | None = None,
    *,
    path: str | Path | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Yield events from the append-only log file."""

    file = _log_file(path)
    if not file.exists():
        return
    with file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:  # pragma: no cover - defensive
                continue
            tick = int(event.get("tick", event.get("step", 0)))
            if tick <= start_tick:
                continue
            if end_tick is not None and tick > end_tick:
                break
            yield event


def log_event(event: dict[str, Any]) -> dict[str, Any]:
    """Log an event to the append-only file and Redpanda if enabled."""

    global _last_hash

    if "trace_hash" in event:
        event = {**event}
        event.pop("trace_hash", None)
    from src.infra.checkpoint import capture_rng_state

    global _seed
    if _seed is None:
        try:
            _seed = random.getstate()[1][0]
        except Exception:  # pragma: no cover - fallback
            _seed = 0

    event = {**event, "rng_state": capture_rng_state(), "seed": _seed}
    if "step" in event and "tick" not in event:
        event["tick"] = event["step"]
    if _last_hash is not None:
        event["prev_hash"] = _last_hash
    event_with_hash = {**event, "trace_hash": compute_trace_hash(event)}
    _last_hash = event_with_hash["trace_hash"]

    # Append to local log file
    try:  # pragma: no cover - best effort
        path = _log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event_with_hash))
            fh.write("\n")
    except Exception:  # pragma: no cover - ignore
        pass

    if os.getenv("ENABLE_REDPANDA", "0") != "1":
        return event_with_hash
    try:  # pragma: no cover - best effort
        payload = json.dumps(event_with_hash).encode("utf-8")
        producer = _get_producer()
        producer.produce(_topic, payload)
        producer.poll(0)
    except Exception as exc:  # pragma: no cover - best effort
        import logging

        logging.getLogger(__name__).debug("Failed to log event: %s", exc)
    return event_with_hash


def fetch_events(
    after_step: int = 0,
    *,
    end_step: int | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieve events from the log after ``after_step``."""

    if os.getenv("ENABLE_REDPANDA", "0") != "1":
        events = list(_iter_file_events(after_step, end_step, path=path))
        return _filter_events(events, after_step=after_step)

    raw_events: list[dict[str, Any]] = []
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
            step = event.get("step", 0)
            if step > after_step and (end_step is None or step <= end_step):
                raw_events.append(event)
    except Exception as exc:  # pragma: no cover - best effort
        import logging

        logging.getLogger(__name__).debug("Failed to fetch events: %s", exc)
    finally:
        try:
            consumer.close()
        except Exception:  # pragma: no cover - ignore
            pass
    return _filter_events(raw_events, after_step=after_step)


def stream_events(
    after_step: int = 0,
    timeout: float = 1.0,
    *,
    end_step: int | None = None,
    path: str | Path | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Yield events from the log or Redpanda until ``timeout`` of inactivity."""

    if os.getenv("ENABLE_REDPANDA", "0") != "1":
        yield from _filter_events(
            _iter_file_events(after_step, end_step, path=path), after_step=after_step
        )
        return

    start = time.time()
    consumer = KafkaConsumer(_consumer_conf)
    consumer.subscribe([_topic])
    last_step = after_step
    last_hash: str | None = None
    try:
        while True:
            msg = consumer.poll(0.1)
            if msg is None:
                if time.time() - start > timeout:
                    break
                continue
            start = time.time()
            if msg.error():
                break
            try:
                event = json.loads(msg.value().decode("utf-8"))
            except Exception:
                continue
            if _is_valid_event(event, last_step, last_hash):
                last_step = event.get("step", last_step)
                last_hash = event.get("trace_hash")
                step = event.get("step", 0)
                if end_step is not None and step > end_step:
                    break
                yield event
    finally:
        try:
            consumer.close()
        except Exception:  # pragma: no cover - ignore
            pass
