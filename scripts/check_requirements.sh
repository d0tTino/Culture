#!/bin/bash
set -euo pipefail

# Ensure pip-tools is installed
command -v pip-compile >/dev/null 2>&1 || { echo "pip-compile not found" >&2; exit 1; }

# Generate requirements to a temp file without writing to disk
TMP_FILE=$(mktemp)
trap 'rm -f "$TMP_FILE"' EXIT

pip-compile --dry-run --no-header --no-annotate --output-file - requirements.in \
  2>&1 | grep -v '^WARNING:' | grep -v 'Dry-run' > "$TMP_FILE"

if ! diff -q requirements.txt "$TMP_FILE" >/dev/null; then
  echo "requirements.txt is out of date. Run 'pip-compile requirements.in' to update." >&2
  diff -u requirements.txt "$TMP_FILE" >&2 || true
  exit 1
fi

echo "requirements.txt is up to date."
