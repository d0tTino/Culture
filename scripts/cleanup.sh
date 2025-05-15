#!/bin/bash

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