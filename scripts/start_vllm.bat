@echo off
REM Start the vLLM OpenAI-compatible API server with recommended defaults.

if not defined VLLM_MODEL set VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
if not defined VLLM_PORT set VLLM_PORT=8001
if not defined VLLM_SWAP_SPACE set VLLM_SWAP_SPACE=16

python -m vllm.entrypoints.openai.api_server ^
  --model "%VLLM_MODEL%" ^
  --port "%VLLM_PORT%" ^
  --swap-space "%VLLM_SWAP_SPACE%"
