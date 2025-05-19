#!/usr/bin/env python
"""
Test script to check for Pydantic Field deprecation warnings.
"""

import warnings

# Set warnings to be as verbose as possible
warnings.filterwarnings("always", category=DeprecationWarning)

# Import the necessary modules
print("Importing langchain.llms.base...")
try:
    print("Imported langchain.llms.base successfully")
except Exception as e:
    print(f"Error importing langchain.llms.base: {e}")

print("Importing langgraph.prebuilt.chat_agent...")
try:
    print("Imported langgraph.prebuilt.chat_agent successfully")
except Exception as e:
    print(f"Error importing langgraph.prebuilt.chat_agent: {e}")

print("Importing chromadb...")
try:
    print("Imported chromadb successfully")
except Exception as e:
    print(f"Error importing chromadb: {e}")

print("Importing discord.py...")
try:
    print("Imported discord.py successfully")
except Exception as e:
    print(f"Error importing discord.py: {e}")

print("Testing complete")
