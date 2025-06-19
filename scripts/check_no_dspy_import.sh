#!/bin/bash
set -e
# Fail if any Python file contains a direct import of the deprecated `dspy` package
pattern='^\s*(from\s+dspy\b|import\s+dspy(\s|$))'
files=$(git ls-files '*.py' | grep '^src/' | grep -v '^src/dspy_ai/__init__\.py$')
if grep -nE "$pattern" $files; then
  echo "Direct 'dspy' imports are forbidden. Use 'dspy_ai' instead." >&2
  exit 1
fi
