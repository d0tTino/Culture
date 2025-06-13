#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible API server with recommended defaults.
set -euo pipefail

# Load environment variables from .env if available
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL=${VLLM_MODEL:-"mistralai/Mistral-7B-Instruct-v0.2"}
PORT=${VLLM_PORT:-8001}
SWAP=${VLLM_SWAP_SPACE:-16}

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --swap-space "${SWAP}"
