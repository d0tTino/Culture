"""Path utilities for the Culture.ai project."""

from __future__ import annotations

import tempfile
from pathlib import Path


def get_temp_dir(prefix: str = "culture_") -> Path:
    """Return a new temporary directory as a :class:`~pathlib.Path`."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` as a directory if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
