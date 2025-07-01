#!/usr/bin/env bash
# Script to install essential dependencies for the Culture.ai "AI Genesis Engine" project.
# Designed for a headless, time-limited environment (e.g., ChatGPT Code Interpreter).
# It installs core libraries for DSPy, LangGraph, Ollama support, Discord bot, and ChromaDB.
# The script favors pre-built binary wheels and CPU-only packages to avoid long build times.

set -e  # Exit immediately on any error to avoid partial installs

# (Optional) Upgrade pip to latest version for better dependency resolution (especially on fresh envs)
python -m pip install --upgrade pip --no-cache-dir

# Install core dependencies:
# - pydantic v2 (data validation, required by LiteLLM)
# - openai (OpenAI API client, needed for OpenAI and proxy usage)
# - litellm (LiteLLM for unified local/remote LLM API, ensures OpenAI>=1.0.0)
# - dspy (DSPy for prompt/program optimization)
# - langgraph (LangChain's LangGraph for agent orchestration)
# - discord.py (Discord bot API wrapper)
# - chromadb (Chroma vector DB for memory storage)
# - python-dotenv (to load configuration from .env files)
echo "Installing core Python packages..."
pip install -U --no-cache-dir --prefer-binary \
    "pydantic>=2.0" \
    openai \
    litellm \
    dspy \
    langgraph \
    discord.py \
    chromadb \
    python-dotenv

# Install sentence-transformers for embeddings.
# This is used for computing text embeddings (e.g., memory vectors) via local models.
# Note: This will pull in Hugging Face Transformers and PyTorch (if not already present).
# It may be slow to install. If running in a very constrained environment, you can skip this
# or install a lighter embedding method. However, it's essential for full simulation functionality.
echo "Installing sentence-transformers (for text embeddings)..."
pip install -U --no-cache-dir --prefer-binary sentence-transformers

# All done. If we reached here, all essential packages are installed.
echo "✅ Setup Complete"
# User-provided custom instructions
