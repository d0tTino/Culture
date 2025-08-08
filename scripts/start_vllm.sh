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

MODEL=${VLLM_MODEL:-${MODEL:-"mistralai/Mistral-7B-Instruct-v0.2"}}
PORT=${VLLM_PORT:-${PORT:-8001}}
SWAP=${VLLM_SWAP_SPACE:-${SWAP:-16}}
GPUS=${VLLM_GPUS:-${GPUS:-0}}
TP_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-${TP_SIZE:-1}}
GPU_UTIL=${VLLM_GPU_MEMORY_UTILIZATION:-${GPU_UTIL:-0.9}}

echo "Starting vLLM server with model '${MODEL}' on port ${PORT} using GPUs ${GPUS}" >&2
echo "Set VLLM_API_BASE=http://localhost:${PORT} to connect" >&2

CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --swap-space "${SWAP}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_UTIL}"
