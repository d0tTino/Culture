# Makefile for Culture project
# Provides helper targets for local development

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

ifeq ($(OS),Windows_NT)
ACTIVATE := .venv\Scripts\activate
VERTICAL_SLICE := scripts\vertical_slice.bat
else
ACTIVATE := .venv/bin/activate
VERTICAL_SLICE := ./scripts/vertical_slice.sh
endif

SNAPSHOTS ?= snapshots
OUTPUT ?= data/traces.jsonl

.PHONY: local-slice
local-slice:
	@if [ -f "$(ACTIVATE)" ]; then \
		source "$(ACTIVATE)"; \
		if [ ! -f ".venv/.deps_installed" ]; then \
			pip install -r requirements.txt -r requirements-dev.txt; \
			touch .venv/.deps_installed; \
		fi; \
	fi; \
	$(VERTICAL_SLICE)

.PHONY: dataset
dataset:
	python - <<-EOF
	from scripts.export_traces import export_latest
	export_latest(directory="$(SNAPSHOTS)", output="$(OUTPUT)")
	EOF

.PHONY: council
council:
	python -m scripts.council_cli --question "$(Q)" $(if $(SHOW_ALL),--show-all)

.PHONY: council-gate-tests
council-gate-tests:
	python -m pytest -m "unit or integration" \
		tests/unit/agents/council/ \
		tests/unit/agents/graphs/test_council_node.py \
		tests/unit/infra/test_council_config.py \
		tests/unit/scripts/test_council_cli.py \
		tests/unit/scripts/test_council_metrics.py \
		tests/integration/agents/council/test_council_du_budget.py
