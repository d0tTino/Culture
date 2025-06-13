#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Load environment variables from .env if available
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

FORMAT=0
if [[ "$1" == "--format" || "$1" == "-f" ]]; then
  FORMAT=1
fi

if [[ $FORMAT -eq 1 ]]; then
  echo "Running Ruff format..."
  ruff format src/ tests/
fi

echo "Running Ruff check..."
ruff check src/ tests/

echo "Running Black..."
black src/ tests/

echo "Running Mypy..."
mypy src/

echo "Linting and formatting complete." 
