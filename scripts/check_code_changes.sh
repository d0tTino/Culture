#!/bin/bash
set -e

# Determine base commit for diff
BASE_SHA="${GITHUB_BASE_SHA:-${GITHUB_EVENT_BEFORE}}"
if [ -z "$BASE_SHA" ]; then
  BASE_SHA=$(git rev-parse HEAD~1)
fi

git fetch --depth=1 origin "$BASE_SHA" >/dev/null 2>&1 || true

# List changed files between base and HEAD
CHANGED_FILES=$(git diff --name-only "$BASE_SHA" HEAD)

# Filter out documentation files
CODE_FILES=$(echo "$CHANGED_FILES" | grep -vE '\.md$|^docs/' || true)

if [ -z "$CODE_FILES" ]; then
  echo "Only documentation files changed."
  echo "CODE_CHANGES=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

# Check for non-comment additions
NON_COMMENT_LINES=$(git diff "$BASE_SHA" HEAD -- $CODE_FILES | grep -E '^\+' | grep -vE '^\+\+' | sed 's/^+//' | grep -vE '^\s*(#|//|\*|"""|$)' || true)

if [ -z "$NON_COMMENT_LINES" ]; then
  echo "Only comments changed."
  echo "CODE_CHANGES=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "Code changes detected."
echo "CODE_CHANGES=true" >> "$GITHUB_OUTPUT"
