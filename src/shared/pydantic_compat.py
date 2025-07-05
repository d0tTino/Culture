from __future__ import annotations

from typing import Any, Callable, cast

try:  # pragma: no cover - prefer pydantic>=2
    from pydantic import field_validator as pyd_field_validator
    from pydantic import model_validator as pyd_model_validator

    _field_validator = pyd_field_validator
    _model_validator = pyd_model_validator
    _PYDANTIC_V2 = True
except Exception:  # pragma: no cover - fallback to pydantic<2
    from pydantic import root_validator, validator

    _model_validator = cast(Callable[..., Any], root_validator)
    _field_validator = cast(Callable[..., Any], validator)
    _PYDANTIC_V2 = False


def field_validator(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper for Pydantic field validators."""
    if not _PYDANTIC_V2:
        mode = kwargs.pop("mode", None)
        if mode == "before":
            kwargs["pre"] = True
    return _field_validator(*args, **kwargs)


def model_validator(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper for Pydantic model validators."""
    if not _PYDANTIC_V2:
        mode = kwargs.pop("mode", None)
        if mode == "before":
            kwargs["pre"] = True
    return _model_validator(*args, **kwargs)

try:
    from pydantic_settings import (
        BaseSettings as _PydanticBaseSettings,
    )
    from pydantic_settings import (
        SettingsConfigDict as _SettingsConfigDict,
    )
except Exception:  # pragma: no cover - optional dependency
    from pydantic import BaseModel as _PydanticBaseSettings

    class _FallbackSettingsConfigDict(dict[str, Any]):
        pass

    _SettingsConfigDict = _FallbackSettingsConfigDict

BaseSettings = _PydanticBaseSettings
SettingsConfigDict = _SettingsConfigDict  # type: ignore[misc]
