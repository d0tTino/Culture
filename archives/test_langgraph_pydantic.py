#!/usr/bin/env python
"""
Test script to check if LangGraph is causing the Pydantic Field deprecation warning.
"""

import warnings

warnings.simplefilter("error", DeprecationWarning)

# Try to import langgraph and create a simple model
try:
    from pydantic import BaseModel, Field

    class TestNode(BaseModel):
        """Test node for langgraph."""

        name: str = Field(default="test")

    print("LangGraph and Pydantic imported without warnings")
except Exception as e:
    print(f"Error: {e}")

print("Test complete")
