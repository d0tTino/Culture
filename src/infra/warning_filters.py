#!/usr/bin/env python
"""
Module to configure warning filters for the application.

This helps suppress warnings from third-party dependencies that we can't fix directly.
"""

import logging
import warnings
from typing import IO, Any, Literal, cast


def configure_warning_filters(apply_filters: bool = True, log_suppressed: bool = False) -> None:
    """Configure warning filters for third-party dependencies.

    Parameters
    ----------
    apply_filters:
        When ``True`` (default) suppress known noisy warnings.
    log_suppressed:
        If ``True``, log warnings that would otherwise be suppressed.
    """
    warnings.resetwarnings()

    if not apply_filters:
        return

    action: Literal["ignore", "once"] = "ignore"
    if log_suppressed:
        action = "once"

        def log_warning(
            message: warnings.WarningMessage | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: IO[str] | None = None,
            line: str | None = None,
        ) -> None:
            logger = logging.getLogger(__name__)
            logger.warning("%s:%s: %s: %s", filename, lineno, category.__name__, message)

        warnings.showwarning = cast(Any, log_warning)
    try:
        # Import the specific warning class from pydantic
        from pydantic.warnings import PydanticDeprecatedSince20

        # Suppress the Pydantic Field deprecation warning from third-party dependencies
        warnings.filterwarnings(
            action,
            message=r"Using extra keyword arguments on `Field` is deprecated.*",
            category=PydanticDeprecatedSince20,
        )
    except ImportError:
        # Fallback if we can't import the specific warning class
        warnings.filterwarnings(
            action,
            message=r"Using extra keyword arguments on `Field` is deprecated.*",
            category=DeprecationWarning,
            module=r"pydantic\.fields",
        )

    # Suppress the audioop deprecation warning from discord.py
    warnings.filterwarnings(
        action,
        message=r"'audioop' is deprecated and slated for removal in Python 3\.13",
        category=DeprecationWarning,
    )


# Apply filters automatically when this module is imported
configure_warning_filters()
