#!/usr/bin/env python3
"""
Cleanup script for temporary ChromaDB directories.
This script removes all temporary ChromaDB directories created during tests.
"""

import argparse
import glob
import shutil


def cleanup_temp_db(dry_run=False):
    """
    Clean up temporary ChromaDB directories.

    Args:
        dry_run: If True, only print the directories that would be removed without removing them.
    """
    # Find all ChromaDB directories except the main persistent one
    chroma_dirs = glob.glob("chroma_db_*")

    if not chroma_dirs:
        print("No temporary ChromaDB directories found.")
        return

    print(f"Found {len(chroma_dirs)} temporary ChromaDB directories:")
    for dir_path in chroma_dirs:
        print(f"  - {dir_path}")

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
    for dir_path in chroma_dirs:
        try:
            shutil.rmtree(dir_path)
            print(f"✓ Removed: {dir_path}")
            removed += 1
        except Exception as e:
            print(f"✗ Failed to remove {dir_path}: {e}")
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
