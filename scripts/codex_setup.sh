#!/usr/bin/env bash
# Bootstrap script for the OpenAI Codex environment
# Installs system packages and Python dependencies for tests and linting

set -euxo pipefail

# Load environment variables from .env if available
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# install required system packages
apt-get update
apt-get install -y git build-essential python3-venv

# create virtual environment if not present
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# activate virtualenv
source .venv/bin/activate

# upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt

# install pre-commit hooks
pre-commit install

echo "Codex environment is ready"
