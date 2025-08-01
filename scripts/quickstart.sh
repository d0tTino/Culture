#!/usr/bin/env bash
# Launch a quick demo with the UI and LLM backend.
# Manual instructions: see README.md#start-vllm or docs/runbook.md#start-vllm
# for required environment variables. After the server starts you can run
# scripts/benchmark_llm.py as a sanity check.
set -euo pipefail

# Load environment variables
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

start_llm() {
  if command -v ollama >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
  elif python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("vllm") else 1)
PY
  then
    echo "Starting vLLM server..."
    scripts/start_vllm.sh &
  else
    echo "Error: neither Ollama nor vLLM is installed." >&2
    exit 1
  fi
}

start_llm
sleep 2

# Start the vertical slice demo in the background
scripts/vertical_slice.sh &

# Launch the culture-ui development server
pnpm --filter culture-ui dev
