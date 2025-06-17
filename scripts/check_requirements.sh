#!/bin/bash
set -euo pipefail

# Ensure pip-tools is installed
command -v pip-compile >/dev/null 2>&1 || { echo "pip-compile not found" >&2; exit 1; }

check_lockfile() {
  local input_file=$1
  local lock_file=$2

  local tmp_file
  tmp_file=$(mktemp)
  trap 'rm -f "$tmp_file"' RETURN

  pip-compile --dry-run --no-header --no-annotate --output-file - "$input_file" \
    2>&1 | grep -v '^WARNING:' | grep -v 'Dry-run' > "$tmp_file"

  if ! diff -q "$lock_file" "$tmp_file" >/dev/null; then
    echo "$lock_file is out of date. Run 'pip-compile $input_file' to update." >&2
    diff -u "$lock_file" "$tmp_file" >&2 || true
    exit 1
  fi

  rm -f "$tmp_file"
  trap - RETURN
  echo "$lock_file is up to date."
}

check_lockfile requirements.in requirements.txt
check_lockfile requirements-dev.in requirements-dev.txt
