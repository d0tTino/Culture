"""Simple Redpanda event logging."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import Any

from opentelemetry import trace

try:  # pragma: no cover - optional dependency
    from confluent_kafka import Consumer as KafkaConsumer
    from confluent_kafka import Producer as KafkaProducer
except Exception:  # pragma: no cover - fallback
    KafkaConsumer = KafkaProducer = Any

try:  # pragma: no cover - optional dependency in tests
    from src.infra.snapshot import compute_trace_hash as _compute_trace_hash
except ImportError:  # pragma: no cover - fallback when tests stub snapshot

    def _compute_trace_hash(data: dict[str, Any]) -> str:
        payload = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


compute_trace_hash = _compute_trace_hash


_AGENT_ACTION_EXPLAIN_DEFAULT: dict[str, Any] = {
    "memories": [],
    "knowledge_board_entries": [],
    "tool_calls": [],
    "rag_summary": None,
}

def _jsonify(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, (tuple, set)):
        return [_jsonify(item) for item in value]
    if value is None:
        return []
    return [_jsonify(value)]


def _sanitize_agent_action(event: dict[str, Any]) -> dict[str, Any]:
    raw_explain = event.get("explain_why")
    sanitized = dict(_AGENT_ACTION_EXPLAIN_DEFAULT)

    if isinstance(raw_explain, Mapping):
        if "memories" in raw_explain:
            sanitized["memories"] = _ensure_list(raw_explain.get("memories"))
        if "knowledge_board_entries" in raw_explain:
            sanitized["knowledge_board_entries"] = [
                str(item) for item in _ensure_list(raw_explain.get("knowledge_board_entries"))
            ]
        if "tool_calls" in raw_explain:
            sanitized["tool_calls"] = _ensure_list(raw_explain.get("tool_calls"))
        if "rag_summary" in raw_explain:
            summary = raw_explain.get("rag_summary")
            if summary is None or isinstance(summary, str):
                sanitized["rag_summary"] = summary
            else:
                sanitized["rag_summary"] = str(summary)

    return {**event, "explain_why": sanitized}


def _sanitize_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") == "agent_action":
        return _sanitize_agent_action(event)
    return event


def candidate_event_logs_for_snapshot(snapshot: str | Path) -> Iterable[Path]:
    """Yield plausible event log paths relative to ``snapshot``."""

    snapshot_path = Path(snapshot)
    directory = snapshot_path.parent

    compression_suffixes = {".zst", ".gz", ".bz2", ".xz"}
    uncompressed_snapshot = snapshot_path
    while uncompressed_snapshot.suffix in compression_suffixes:
        uncompressed_snapshot = uncompressed_snapshot.with_suffix("")

    stem = uncompressed_snapshot.stem
    if stem.endswith(".json"):
        stem = stem[:-5]
    candidates = [
        directory / "events.jsonl",
        directory / "event_log.jsonl",
        directory / f"{stem}.events.jsonl",
        directory / f"{stem}.event_log.jsonl",
        uncompressed_snapshot.with_suffix(".jsonl"),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def resolve_replay_event_log(
    snapshot: str | Path, explicit: str | Path | None = None
) -> Path | None:
    """Determine which event log file should be used for replay."""

    if explicit:
        return Path(explicit)
    for candidate in candidate_event_logs_for_snapshot(snapshot):
        if candidate.exists():
            return candidate
    return None

tracer = trace.get_tracer(__name__)

_broker = os.getenv("REDPANDA_BROKER", "localhost:9092")
_topic = os.getenv("REDPANDA_TOPIC", "culture.events")

_producer: Any | None = None
_last_hash: str | None = None
_seed: int | None = None
_seed_cache: dict[Path, int] = {}
_header_written: set[Path] = set()


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
    file = _log_file(path)
    resolved = _resolved_path(file)
    cached = _seed_cache.get(resolved)
    if cached is not None:
        return cached

    header = get_log_header(path)
    seed = header.get("seed") if isinstance(header, dict) else None
    if isinstance(seed, int):
        _seed_cache[resolved] = seed
        _seed = seed
        return seed
    return _seed


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


def _resolved_path(path: Path) -> Path:
    """Return a stable cache key for ``path``."""

    try:
        return path.resolve()
    except Exception:
        return Path(os.path.abspath(path))


def _get_or_create_seed(resolved: Path) -> int:
    """Return the cached seed for ``resolved`` or generate a new one."""

    global _seed
    cached = _seed_cache.get(resolved)
    if cached is not None:
        return cached

    if _seed is None:
        try:
            _seed = random.getstate()[1][0]
        except Exception:  # pragma: no cover - fallback
            _seed = 0

    _seed_cache[resolved] = _seed
    return _seed


def _rehydrate_log_state(path: str | Path | None = None) -> None:
    """Populate cached state from the existing log file if available."""

    global _last_hash, _seed
    needs_hash = _last_hash is None
    file = _log_file(path)
    resolved = _resolved_path(file)
    needs_seed = resolved not in _seed_cache
    if not needs_hash and not needs_seed:
        return

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
                    _seed_cache[resolved] = header_seed
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
                    _seed_cache[resolved] = seed
    except Exception:  # pragma: no cover - ignore
        pass


def _ensure_header(path: str | Path | None = None) -> None:
    """Write the log header containing the simulation seed if missing."""
    global _header_written, _seed
    file = _log_file(path)
    resolved = _resolved_path(file)
    if resolved in _header_written:
        return

    seed_value = _get_or_create_seed(resolved)
    header = {"type": "header", "seed": seed_value}
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
                existing_seed = existing.get("seed")
                if isinstance(existing_seed, int):
                    _seed = existing_seed
                    _seed_cache[resolved] = existing_seed
                _header_written.add(resolved)
                return
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists() or file.stat().st_size == 0:
            with file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(header))
                fh.write("\n")
        _header_written.add(resolved)
        if os.getenv("ENABLE_REDPANDA", "0") == "1":
            try:
                producer = _get_producer()
                producer.produce(_topic, json.dumps(header).encode("utf-8"))
                producer.poll(0)
            except Exception:
                pass
    except Exception:  # pragma: no cover - ignore
        pass


def _rehydrate_last_hash(path: Path) -> None:
    """Populate ``_last_hash`` from the most recent event on disk."""

    global _last_hash
    if _last_hash is not None or not path.exists():
        return
    try:  # pragma: no cover - best effort
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
            if pos <= 0:
                return
            buffer = bytearray()
            while pos > 0:
                pos -= 1
                fh.seek(pos)
                char = fh.read(1)
                if char == b"\n":
                    if not buffer:
                        continue
                    line_bytes = bytes(reversed(buffer)).strip()
                    buffer.clear()
                    if not line_bytes:
                        continue
                    try:
                        line = line_bytes.decode("utf-8")
                        event = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(event, dict) and event.get("type") != "header":
                        last_hash = event.get("trace_hash")
                        if isinstance(last_hash, str):
                            _last_hash = last_hash
                        return
                    continue
                buffer.append(char[0])
            if buffer:
                line_bytes = bytes(reversed(buffer)).strip()
                if not line_bytes:
                    return
                try:
                    line = line_bytes.decode("utf-8")
                    event = json.loads(line)
                except Exception:
                    return
                if isinstance(event, dict) and event.get("type") != "header":
                    last_hash = event.get("trace_hash")
                    if isinstance(last_hash, str):
                        _last_hash = last_hash
    except Exception:
        pass


_STEP_EQ_ALLOWED_TYPES = {"human_command"}


def _is_valid_event(event: dict[str, Any], last_step: int, last_hash: str | None) -> bool:
    """Check ``event`` ordering and integrity."""
    step = event.get("step", 0)
    event_type = event.get("type")
    if step <= last_step and event_type not in _STEP_EQ_ALLOWED_TYPES:
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
            if ev.get("type") not in _STEP_EQ_ALLOWED_TYPES:
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
    resolved = _resolved_path(path)
    _rehydrate_log_state(path)
    _ensure_header(path)

    event = _sanitize_event_payload(event)

    with tracer.start_as_current_span("event_log.log_event") as span:
        span.set_attribute("event.type", event.get("type"))
        span.set_attribute("step", event.get("step"))

        span.set_attribute("log.file_path", str(path))
        _rehydrate_last_hash(path)

        if "trace_hash" in event:
            event = {**event}
            event.pop("trace_hash", None)
        from src.infra.checkpoint import capture_rng_state

        seed_value = _get_or_create_seed(resolved)

        event = {**event, "rng_state": capture_rng_state(), "seed": seed_value}
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
