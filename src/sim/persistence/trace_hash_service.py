from __future__ import annotations

import hashlib
import json
from typing import Any


class TraceHashService:
    """Stable trace hash utilities used by replay/snapshot services."""

    @staticmethod
    def compute(data: dict[str, Any]) -> str:
        payload = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
