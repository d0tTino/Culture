#!/usr/bin/env bash
set -euo pipefail

profile="${PROFILE:-public_persistent}"
if [[ "$profile" != "public_persistent" ]]; then
  echo "[warn] PROFILE='$profile' overridden to public_persistent for this launcher."
  profile="public_persistent"
fi

errors=()

require_cmd() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    errors+=("Missing '$cmd'. $hint")
  fi
}

require_cmd python "Install Python 3.11+ and ensure it is on PATH."

python - <<'PY' || errors+=("Python package validation failed. Install project deps with: pip install -r requirements.txt -r requirements-dev.txt")
import importlib
import sys
required = ["fastapi", "uvicorn", "httpx", "pydantic"]
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    print("Missing python packages:", ", ".join(missing))
    sys.exit(1)
print("Python dependencies look good.")
PY

: "${LLM_API_BASE:=http://localhost:8001/v1}"
: "${REDPANDA_BROKER:=localhost:9092}"
: "${GRAPH_DB_URI:=bolt://localhost:7687}"

if ! curl -fsS "${LLM_API_BASE%/}/models" >/dev/null 2>&1; then
  errors+=("LLM endpoint not reachable at $LLM_API_BASE/models. Start vLLM/Ollama and export LLM_API_BASE.")
fi

if [[ ${#errors[@]} -gt 0 ]]; then
  echo "Public persistent profile preflight failed:" >&2
  for err in "${errors[@]}"; do
    echo " - $err" >&2
  done
  exit 1
fi

echo "Preflight passed. Starting Culture with PROFILE=$profile"
exec env PROFILE="$profile" python -m src.app "$@"
