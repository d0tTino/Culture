#!/usr/bin/env python3
"""
Cleanup script for temporary ChromaDB directories.
This script removes all temporary ChromaDB directories created during tests.
"""

import argparse
import shutil
import tempfile
from pathlib import Path


def cleanup_temp_db(dry_run: bool = False) -> None:
    """
    Clean up temporary ChromaDB directories.

    Args:
        dry_run: If True, only print the directories that would be removed without removing them.
    """
    # Search the system temp directory for ChromaDB directories
    base = Path(tempfile.gettempdir())
    chroma_dirs = list(base.glob("chroma_db_*"))

    if not chroma_dirs:
        print("No temporary ChromaDB directories found.")
        return

    print(f"Found {len(chroma_dirs)} temporary ChromaDB directories:")
    for path in chroma_dirs:
        print(f"  - {path}")

    if dry_run:
        print("\nDRY RUN: No directories were removed.")
        return

    # Confirm before deletion
    confirm = input(
        f"\nAre you sure you want to remove these {len(chroma_dirs)} directories? (y/n): "
    )
    if confirm.lower() != "y":
        print("Cleanup aborted.")
        return

    # Remove directories
    removed = 0
    failed = 0
    for path in chroma_dirs:
        try:
            # Iterate contents to ensure proper permissions before deletion
            for _ in path.iterdir():
                pass
            shutil.rmtree(path)
            print(f"✓ Removed: {path}")
            removed += 1
        except Exception as e:
            print(f"✗ Failed to remove {path}: {e}")
            failed += 1

    print(f"\nCleanup completed: {removed} directories removed, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up temporary ChromaDB directories")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print directories that would be removed without removing them",
    )
    args = parser.parse_args()

    cleanup_temp_db(dry_run=args.dry_run)
