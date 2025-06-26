"""Utility functions for embeddings used across agents."""

from __future__ import annotations

import hashlib


def compute_embedding(text: str, dim: int = 8) -> list[float]:
    """Return a deterministic embedding vector for ``text``."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    segment_len = len(digest) // dim
    return [
        int(digest[i * segment_len : (i + 1) * segment_len], 16) / (16**segment_len)
        for i in range(dim)
    ]
