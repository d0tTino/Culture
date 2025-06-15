#!/usr/bin/env bash
# Run the walking vertical slice demo. Activates venv if present.

set -euo pipefail

# Load environment variables from .env if available
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Activate either `venv` or `.venv` if present
if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m examples.walking_vertical_slice
