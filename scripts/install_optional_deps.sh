#!/usr/bin/env bash
# Install optional dependencies required for certain tests and features.
set -euo pipefail

# Load environment variables from .env if available
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

pip install chromadb weaviate-client langgraph
