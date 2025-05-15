#!/usr/bin/env python
"""
Test script to check for Pydantic Field deprecation warnings.
"""

import warnings
import inspect
import traceback
from traceback_with_variables import activate_by_import

# Set warnings to be as verbose as possible
warnings.filterwarnings("always", category=DeprecationWarning)

# Import the necessary modules
print("Importing langchain.llms.base...")
try:
    from langchain.llms.base import BaseLLM
    print("Imported langchain.llms.base successfully")
except Exception as e:
    print(f"Error importing langchain.llms.base: {e}")

print("Importing langgraph.prebuilt.chat_agent...")
try:
    from langgraph.prebuilt.chat_agent import ChatAgent
    print("Imported langgraph.prebuilt.chat_agent successfully")
except Exception as e:
    print(f"Error importing langgraph.prebuilt.chat_agent: {e}")

print("Importing chromadb...")
try:
    import chromadb
    print("Imported chromadb successfully")
except Exception as e:
    print(f"Error importing chromadb: {e}")

print("Importing discord.py...")
try:
    import discord
    print("Imported discord.py successfully")
except Exception as e:
    print(f"Error importing discord.py: {e}")

print("Testing complete") 