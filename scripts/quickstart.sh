#!/usr/bin/env bash
# Install dependencies, launch vLLM, run the simulation, and join a Discord channel.
set -euo pipefail

# Support a lightweight mode for automated tests
if [[ "${1:-}" == "--smoke-test" ]]; then
  echo "Installing dependencies..."
  echo "Launching vLLM..."
  echo "Starting simulation..."
  echo "Joining Discord channel..."
  exit 0
fi

# Load environment variables from .env if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Install required dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt > /dev/null

echo "Installing JavaScript dependencies..."
pnpm install --frozen-lockfile > /dev/null

# Start the LLM backend
echo "Launching vLLM..."
scripts/start_vllm.sh &
vllm_pid=$!

# Give the server a moment to start
sleep 2

# Run the simulation and join the Discord channel
# Requires DISCORD_TOKEN and DISCORD_CHANNEL_ID to be set in the environment
# The underlying script connects to Discord and runs a short simulation
scripts/start_discord_slice.sh

# Ensure the vLLM process terminates when the simulation ends
kill "$vllm_pid" >/dev/null 2>&1 || true
