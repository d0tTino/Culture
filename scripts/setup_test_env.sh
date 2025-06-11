#!/usr/bin/env bash
# Set up a Python virtual environment and install dependencies for Culture.ai tests

set -euo pipefail

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

if ! python -c "import xdist" >/dev/null 2>&1; then
    echo "pytest-xdist is required but not installed; installing..." >&2
    pip install pytest-xdist
fi

echo "Environment ready. Run pytest with $VENV_DIR/bin/python -m pytest"
