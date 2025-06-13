#!/bin/bash
# Aggressive cleanup for Python/AI project
set -e

# Load environment variables from .env if available
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
rm -rf __pycache__ .mypy_cache .ruff_cache .pytest_cache htmlcov logs temp chroma_db scripts/temp data/logs archives/__pycache__ scripts/archive/__pycache__ src/agents/__pycache__ src/agents/core/__pycache__ src/agents/memory/__pycache__ src/agents/dspy_programs/__pycache__ tests/__pycache__ tests/unit/__pycache__ tests/integration/__pycache__
find . -name '*.log' -delete
find . -name '*.txt' -delete
find . -name '*.out' -delete
find . -name '*.pyc' -delete
find . -name '*.pyo' -delete
find . -name '*.pyd' -delete
find . -name '*.pdb' -delete
find . -name '.coverage*' -delete
find . -name 'final_test_suite_output.txt' -delete
find . -name 'pytest_*.txt' -delete
find . -name 'coverage_*.txt' -delete
find archives -name '*.zip' -delete

echo "Cleaning up temporary directories..."

# Check if temp directory exists, if not create it
mkdir -p temp

# Move test directories to temp folder
echo "Moving temp test directories..."
[ -d test_memory_utility_score_* ] && mv test_memory_utility_score_* temp/
[ -d test_mus_pruning_* ] && mv test_mus_pruning_* temp/
[ -d temp_extract ] && mv temp_extract temp/
[ -d __pycache__ ] && mv __pycache__ temp/

# Clean Python cache files
echo "Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "Cleanup complete!" 
