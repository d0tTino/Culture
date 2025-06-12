#!/usr/bin/env bash
# Bootstrap script for the OpenAI Codex environment
# Installs system packages and Python dependencies for tests and linting

set -euxo pipefail

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
