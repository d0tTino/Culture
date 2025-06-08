#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

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
mypy src/ tests/

echo "Linting and formatting complete." 
