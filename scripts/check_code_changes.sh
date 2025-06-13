#!/bin/bash
set -e

# Determine base commit for diff
BASE_SHA="${GITHUB_BASE_SHA:-${GITHUB_EVENT_BEFORE}}"
if [ -z "$BASE_SHA" ]; then
  if git rev-parse HEAD~1 >/dev/null 2>&1; then
    BASE_SHA=$(git rev-parse HEAD~1)
  else
    BASE_SHA=$(git rev-parse HEAD)
  fi
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
DIFF_OUTPUT=$(git diff "$BASE_SHA" HEAD -- $CODE_FILES)

# Extract added lines that are not comments or docstrings
NON_COMMENT_LINES=$(echo "$DIFF_OUTPUT" | awk '
  BEGIN { in_triple = 0 }
  /^@@/ { next }
  /^\+\+\+|^---/ { next }
  {
    sign = substr($0, 1, 1)
    if (sign == "+" || sign == "-" || sign == " ") {
      line = substr($0, 2)
    } else {
      line = $0
    }

    quote_count = gsub(/"""/, "", line)
    doc_line = (in_triple || quote_count > 0)

    if (sign == "+") {
      if (!doc_line && line !~ /^\s*(#|\/\/|\*|$)/) {
        print line
      }
    }

    if (quote_count % 2 == 1) {
      in_triple = !in_triple
    }
  }
')

if [ -z "$NON_COMMENT_LINES" ]; then
  echo "Only comments changed."
  echo "CODE_CHANGES=false" >> "$GITHUB_OUTPUT"
  exit 0
fi

echo "Code changes detected."
echo "CODE_CHANGES=true" >> "$GITHUB_OUTPUT"
