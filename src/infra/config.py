"""Configuration utilities for the Culture project."""
from __future__ import annotations

import importlib
import logging
from typing import Any, cast

from .settings import ConfigSettings, settings

logger = logging.getLogger(__name__)

# Allow runtime overrides in tests
CONFIG_OVERRIDES: dict[str, Any] = {}

REQUIRED_CONFIG_KEYS = ["REDPANDA_BROKER", "OPA_URL"]


def load_config(*, validate_required: bool = True) -> dict[str, Any]:
    """Reload configuration from environment variables."""
    global settings
    new_settings = ConfigSettings()
    if validate_required:
        missing = [k for k in REQUIRED_CONFIG_KEYS if not getattr(new_settings, k, None)]
        if missing:
            raise RuntimeError(
                "Missing mandatory configuration keys: " + ", ".join(missing)
            )
    settings = new_settings
    try:
        data = settings.model_dump()
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        data = settings.dict()
    return cast(dict[str, Any], data)

def get_config(key: str | None = None) -> Any:
    """Return a configuration value from :class:`ConfigSettings`."""
    if key is None:
        try:
            return settings.model_dump()
        except AttributeError:  # pragma: no cover - pydantic v1
            return settings.dict()
    return getattr(settings, key)


def get(setting_name: str, default: str | None = None) -> object:
    """Retrieve a setting value with a fallback default."""
    return getattr(settings, setting_name, default)


RELATIONSHIP_LABELS = {
    (-1.0, -0.7): "Hostile",
    (-0.7, -0.4): "Negative",
    (-0.4, -0.1): "Cautious",
    (-0.1, 0.1): "Neutral",
    (0.1, 0.4): "Cordial",
    (0.4, 0.7): "Positive",
    (0.7, 1.0): "Allied",
}

def get_relationship_label(score: float) -> str:
    """Return a descriptive relationship label for ``score``."""
    for (min_val, max_val), label in RELATIONSHIP_LABELS.items():
        if min_val <= score <= max_val:
            return label
    return "Neutral"


def get_redis_config() -> dict[str, object]:
    """Return Redis connection details as a dictionary."""
    return {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "db": settings.REDIS_DB,
        "password": getattr(settings, "REDIS_PASSWORD", None),
    }


def get_config_value_with_override(
    key: str, default: Any = None, module_name: str = "src.infra.config"
) -> Any:
    """Fetch a config value, checking overrides first."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.error("Failed to import configuration module: %s", module_name)
        return default

    if CONFIG_OVERRIDES.get(key) is not None:
        return CONFIG_OVERRIDES[key]

    return getattr(module, str(key), default)


def __getattr__(name: str) -> Any:
    return getattr(settings, name)
