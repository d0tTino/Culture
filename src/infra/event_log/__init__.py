"""Simple Redpanda event logging."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

from opentelemetry import trace

try:  # pragma: no cover - optional dependency
    from confluent_kafka import Consumer as KafkaConsumer
    from confluent_kafka import Producer as KafkaProducer
except Exception:  # pragma: no cover - fallback
    KafkaConsumer = KafkaProducer = Any

from src.infra.snapshot import compute_trace_hash

tracer = trace.get_tracer(__name__)

_broker = os.getenv("REDPANDA_BROKER", "localhost:9092")
_topic = os.getenv("REDPANDA_TOPIC", "culture.events")

_producer: Any | None = None
_last_hash: str | None = None
_seed: int | None = None
_header_written: bool = False


def set_seed(seed: int) -> None:
    """Inject a stable seed value for event logging."""
    global _seed
    _seed = seed


def get_log_header(path: str | Path | None = None) -> dict[str, Any]:
    """Return the header information from the event log if present."""
    file = _log_file(path)
    if not file.exists():
        return {}
    try:
        with file.open("r", encoding="utf-8") as fh:
            first = fh.readline().strip()
        header = json.loads(first)
        if isinstance(header, dict) and header.get("type") == "header":
            return header
    except Exception:  # pragma: no cover - defensive
        pass
    return {}


def get_seed(path: str | Path | None = None) -> int | None:
    """Return the seed from the event log header if available."""
    global _seed
    if _seed is not None:
        return _seed
    header = get_log_header(path)
    seed = header.get("seed") if isinstance(header, dict) else None
    if isinstance(seed, int):
        _seed = seed
        return seed
    return None


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


def _ensure_header(path: str | Path | None = None) -> None:
    """Write the log header containing the simulation seed if missing."""
    global _header_written, _seed
    if _header_written:
        return
    file = _log_file(path)
    if _seed is None:
        try:
            _seed = random.getstate()[1][0]
        except Exception:  # pragma: no cover - fallback
            _seed = 0
    header = {"type": "header", "seed": _seed}
    try:  # pragma: no cover - best effort
        if file.exists():
            with file.open("r", encoding="utf-8") as fh:
                first = fh.readline().strip()
            try:
                existing = json.loads(first)
            except Exception:
                existing = {}
            if existing.get("type") == "header" and "seed" in existing:
                # Populate the cached seed from the existing header to ensure
                # subsequent ``log_event`` calls embed the same seed value.
                if _seed is None and isinstance(existing.get("seed"), int):
                    _seed = existing["seed"]
                _header_written = True
                return
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists() or file.stat().st_size == 0:
            with file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(header))
                fh.write("\n")
        _header_written = True
        if os.getenv("ENABLE_REDPANDA", "0") == "1":
            try:
                producer = _get_producer()
                producer.produce(_topic, json.dumps(header).encode("utf-8"))
                producer.poll(0)
            except Exception:
                pass
    except Exception:  # pragma: no cover - ignore
        pass


def _is_valid_event(event: dict[str, Any], last_step: int, last_hash: str | None) -> bool:
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
    _ensure_header()

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
    source = "redpanda" if os.getenv("ENABLE_REDPANDA", "0") == "1" else "file"
    with tracer.start_as_current_span("event_log.fetch_events") as span:
        span.set_attribute("event.source", source)
        span.set_attribute("tick.start", after_step)
        if end_step is not None:
            span.set_attribute("tick.end", end_step)

        if source == "file":
            events = list(_iter_file_events(after_step, end_step, path=path))
            return _filter_events(events, after_step=after_step)

        raw_events: list[dict[str, Any]] = []
        consumer: Any | None = None
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
                if consumer is not None:
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
    source = "redpanda" if os.getenv("ENABLE_REDPANDA", "0") == "1" else "file"
    with tracer.start_as_current_span("event_log.stream_events") as span:
        span.set_attribute("event.source", source)
        span.set_attribute("tick.start", after_step)
        if end_step is not None:
            span.set_attribute("tick.end", end_step)

        if source == "file":
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
