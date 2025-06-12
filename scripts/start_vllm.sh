#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible API server with recommended defaults.
set -euo pipefail

MODEL=${VLLM_MODEL:-"mistralai/Mistral-7B-Instruct-v0.2"}
PORT=${VLLM_PORT:-8000}
SWAP=${VLLM_SWAP_SPACE:-16}

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --swap-space "${SWAP}"
