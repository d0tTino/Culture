from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

MIGRATION_WINDOW_END = date.fromisoformat(
    os.getenv("LEGACY_COMMAND_MIGRATION_WINDOW_END", "2026-12-31")
)


@dataclass(frozen=True)
class LegacyAdapterTelemetry:
    adapter: str
    fields: tuple[str, ...]
    context_source: str | None
    sender_id: str | None
    correlation_id: str | None

    def asdict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "fields": list(self.fields),
            "context_source": self.context_source,
            "sender_id": self.sender_id,
            "correlation_id": self.correlation_id,
            "migration_window_end": MIGRATION_WINDOW_END.isoformat(),
        }


_LEGACY_ADAPTER_HITS: list[LegacyAdapterTelemetry] = []


def reset_legacy_adapter_hits() -> None:
    _LEGACY_ADAPTER_HITS.clear()


def get_legacy_adapter_hits() -> list[LegacyAdapterTelemetry]:
    return list(_LEGACY_ADAPTER_HITS)


def assert_no_expired_legacy_adapter_usage(*, today: date | None = None) -> None:
    current_day = today or date.today()
    if current_day <= MIGRATION_WINDOW_END:
        return
    if _LEGACY_ADAPTER_HITS:
        raise AssertionError(
            "Legacy command adapter usage remains after migration window: "
            + ", ".join(hit.adapter for hit in _LEGACY_ADAPTER_HITS)
        )


def normalize_legacy_command_payload(
    payload: Mapping[str, Any],
    *,
    adapter: str = "legacy_payload_normalizer",
    context_source: str | None = None,
    sender_id: str | None = None,
) -> dict[str, Any]:
    data = dict(payload)
    deprecated_fields: list[str] = []
    if "content" in data and "text" not in data:
        data["text"] = data["content"]
        deprecated_fields.append("content")
    if "prompt" in data and "scope" not in data:
        data["scope"] = data["prompt"]
        deprecated_fields.append("prompt")
    if "command_type" in data and "intent" not in data and "type" not in data:
        data["intent"] = data["command_type"]
        deprecated_fields.append("command_type")
    if "type" in data and "intent" not in data:
        data["intent"] = data["type"]
        deprecated_fields.append("type")

    if deprecated_fields:
        telemetry = LegacyAdapterTelemetry(
            adapter=adapter,
            fields=tuple(sorted(deprecated_fields)),
            context_source=context_source,
            sender_id=sender_id,
            correlation_id=(
                str(data.get("correlation_id")) if data.get("correlation_id") is not None else None
            ),
        )
        _LEGACY_ADAPTER_HITS.append(telemetry)
        logger.warning("legacy_command_adapter_hit", extra={"telemetry": telemetry.asdict()})
    return data
