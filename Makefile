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
