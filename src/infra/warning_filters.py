#!/usr/bin/env python
"""
Module to configure warning filters for the application.

This helps suppress warnings from third-party dependencies that we can't fix directly.
"""

import warnings
import re

def configure_warning_filters():
    """
    Configure warning filters to suppress specific warnings from dependencies.
    
    This is called during application startup to reduce noise from warnings
    that come from third-party dependencies.
    """
    try:
        # Import the specific warning class from pydantic
        from pydantic.warnings import PydanticDeprecatedSince20
        
        # Suppress the Pydantic Field deprecation warning from third-party dependencies
        warnings.filterwarnings(
            "ignore", 
            message=r"Using extra keyword arguments on `Field` is deprecated.*",
            category=PydanticDeprecatedSince20
        )
    except ImportError:
        # Fallback if we can't import the specific warning class
        warnings.filterwarnings(
            "ignore", 
            message=r"Using extra keyword arguments on `Field` is deprecated.*",
            category=DeprecationWarning,
            module=r"pydantic\.fields"
        )
    
    # Suppress the audioop deprecation warning from discord.py
    warnings.filterwarnings(
        "ignore",
        message=r"'audioop' is deprecated and slated for removal in Python 3\.13",
        category=DeprecationWarning
    )

# Apply filters automatically when this module is imported
configure_warning_filters() 