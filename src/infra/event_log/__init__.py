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


def _rehydrate_log_state(path: str | Path | None = None) -> None:
    """Populate cached state from the existing log file if available."""

    global _last_hash, _seed
    needs_hash = _last_hash is None
    needs_seed = _seed is None
    if not needs_hash and not needs_seed:
        return

    file = _log_file(path)
    if not file.exists():
        return

    try:  # pragma: no cover - best effort
        with file.open("r", encoding="utf-8") as fh:
            header_line = fh.readline().strip()
            header: dict[str, Any] | None = None
            if header_line:
                try:
                    maybe_header = json.loads(header_line)
                except Exception:
                    maybe_header = None
                if isinstance(maybe_header, dict) and maybe_header.get("type") == "header":
                    header = maybe_header

            if needs_seed and header is not None:
                header_seed = header.get("seed")
                if isinstance(header_seed, int):
                    _seed = header_seed
                    needs_seed = False

            last_event: dict[str, Any] | None = None
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                last_event = event

            if last_event is None:
                return

            if needs_hash:
                trace_hash = last_event.get("trace_hash")
                if isinstance(trace_hash, str):
                    _last_hash = trace_hash
                    needs_hash = False

            if needs_seed:
                seed = last_event.get("seed")
                if isinstance(seed, int):
                    _seed = seed
    except Exception:  # pragma: no cover - ignore
        pass


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
    path = _log_file()
    _rehydrate_log_state(path)
    _ensure_header(path)

    with tracer.start_as_current_span("event_log.log_event") as span:
        span.set_attribute("event.type", event.get("type"))
        span.set_attribute("step", event.get("step"))

        span.set_attribute("log.file_path", str(path))

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
        serialized_event = json.dumps(event_with_hash)
        persisted_event = json.loads(serialized_event)
        _last_hash = persisted_event["trace_hash"]

        # Append to local log file
        try:  # pragma: no cover - best effort
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(serialized_event)
                fh.write("\n")
        except Exception:  # pragma: no cover - ignore
            pass

        if os.getenv("ENABLE_REDPANDA", "0") != "1":
            return persisted_event
        try:  # pragma: no cover - best effort
            payload = serialized_event.encode("utf-8")
            producer = _get_producer()
            producer.produce(_topic, payload)
            producer.poll(0)
        except Exception as exc:  # pragma: no cover - best effort
            import logging

            logging.getLogger(__name__).debug("Failed to log event: %s", exc)
        return persisted_event


def log_misbehavior(event: dict[str, Any]) -> dict[str, Any]:
    """Log a misbehavior event and record a dedicated span."""

    mis_event = {**event, "type": "misbehavior"}
    with tracer.start_as_current_span("event_log.misbehavior") as span:
        span.set_attribute("event.type", "misbehavior")
        span.set_attribute("step", mis_event.get("step"))
        logged = log_event(mis_event)
        span.set_attribute("seed", logged.get("seed"))
        span.set_attribute("prev_hash", logged.get("prev_hash"))
        span.set_attribute("trace_hash", logged.get("trace_hash"))
        return logged


def fetch_events(
    after_step: int = 0,
    *,
    end_step: int | None = None,
    path: str | Path | None = None,
    event_type: str | None = None,
    include_misbehavior: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve events from the log after ``after_step``.

    Specify ``event_type`` to return only events of the given type. By default,
    misbehavior events are filtered out. Set ``include_misbehavior`` to
    ``True`` to include them in the results.
    """
    source = "redpanda" if os.getenv("ENABLE_REDPANDA", "0") == "1" else "file"
    with tracer.start_as_current_span("event_log.fetch_events") as span:
        span.set_attribute("event.source", source)
        span.set_attribute("tick.start", after_step)
        if end_step is not None:
            span.set_attribute("tick.end", end_step)

        if source == "file":
            events = list(_iter_file_events(after_step, end_step, path=path))
            events = _filter_events(events, after_step=after_step)
            if event_type is not None:
                events = [ev for ev in events if ev.get("type") == event_type]
            elif not include_misbehavior:
                events = [ev for ev in events if ev.get("type") != "misbehavior"]
            return events

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
        events = _filter_events(raw_events, after_step=after_step)
        if event_type is not None:
            events = [ev for ev in events if ev.get("type") == event_type]
        elif not include_misbehavior:
            events = [ev for ev in events if ev.get("type") != "misbehavior"]
        return events


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


def store_replay_slice(
    start_step: int, end_step: int, directory: str | Path | None = None
) -> Path:
    """Persist events between ``start_step`` and ``end_step`` inclusive.

    The slice is saved as ``replay_<start>_<end>.jsonl`` in ``directory`` and
    can later be used for replaying portions of the simulation.
    """

    events = fetch_events(after_step=start_step - 1, end_step=end_step)
    dest_dir = Path(directory) if directory is not None else _log_file().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"replay_{start_step}_{end_step}.jsonl"
    seed = get_seed()
    with out.open("w", encoding="utf-8") as fh:
        header: dict[str, Any] = {"type": "header"}
        if seed is not None:
            header["seed"] = seed
        fh.write(json.dumps(header))
        fh.write("\n")
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")
    return out
