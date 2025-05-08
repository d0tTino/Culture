# Utility Scripts for Culture.ai

This directory contains utility scripts for managing and maintaining the Culture.ai project.

## Available Scripts

### `cleanup_temp_db.py`

Cleans up temporary ChromaDB directories that are created during tests.

Usage:
```bash
# Show directories that would be removed without removing them
python scripts/cleanup_temp_db.py --dry-run

# Actually remove the directories (will prompt for confirmation)
python scripts/cleanup_temp_db.py
``` 