#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running Ruff format..."
ruff format src/ tests/

echo "Running Ruff check..."
ruff check src/ tests/

echo "Running Black..."
black src/ tests/

echo "Running Mypy..."
mypy src/ tests/

echo "Linting and formatting complete." 
