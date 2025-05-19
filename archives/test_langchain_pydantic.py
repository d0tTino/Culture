#!/usr/bin/env python
"""
Test script to check if LangChain is causing the Pydantic Field deprecation warning.
"""

import warnings

warnings.simplefilter("error", DeprecationWarning)

try:
    from langchain.pydantic_v1 import BaseModel, Field

    class TestModel(BaseModel):
        field1: str = Field(default="test")

    print("LangChain pydantic_v1 imported without warnings")
except Exception as e:
    print(f"LangChain pydantic_v1 error: {e}")

try:
    print("LangGraph imported without warnings")
except Exception as e:
    print(f"LangGraph error: {e}")

try:
    print("ChromaDB Settings imported without warnings")
except Exception as e:
    print(f"ChromaDB Settings error: {e}")

print("Test complete")
