#!/usr/bin/env python
"""
Utility script to clean up test database directories that accumulate from running tests.
This script identifies and removes test database directories to recover disk space.
"""

import os
import shutil
import re
import argparse
from datetime import datetime

# Regular expressions to match test database directories
TEST_DIR_PATTERNS = [
    r'test_chroma_pruning_[a-z0-9]+',
    r'test_chroma_dbs/test_.*',
    r'test_full_memory_pipeline_[a-z0-9]+',
    r'test_memory_utility_score_[a-z0-9]+',
    r'test_chroma_pruning_[a-z0-9]+',
    r'chroma_benchmark_.*'
]

def find_test_directories(base_dir='.'):
    """Find all test database directories that match our patterns."""
    test_dirs = []
    
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            # Skip .git and other hidden directories
            if dir_name.startswith('.'):
                continue
                
            # Check if the directory name matches any of our patterns
            for pattern in TEST_DIR_PATTERNS:
                if re.fullmatch(pattern, dir_name) or re.fullmatch(pattern, full_path[2:]):  # [2:] to skip './'
                    test_dirs.append(full_path)
                    break
    
    return test_dirs

def clean_test_directories(dirs, dry_run=True):
    """Remove the identified test directories."""
    removed = 0
    failed = 0
    total_size = 0
    
    for dir_path in dirs:
        try:
            # Calculate directory size if available
            dir_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(dir_path)
                for filename in filenames
            )
            
            total_size += dir_size
            size_mb = dir_size / (1024 * 1024)
            
            if dry_run:
                print(f"Would remove: {dir_path} ({size_mb:.2f} MB)")
            else:
                print(f"Removing: {dir_path} ({size_mb:.2f} MB)")
                shutil.rmtree(dir_path)
            removed += 1
        except Exception as e:
            print(f"Error removing {dir_path}: {e}")
            failed += 1
    
    total_size_mb = total_size / (1024 * 1024)
    action = "Would free" if dry_run else "Freed"
    print(f"\n{action} {total_size_mb:.2f} MB of disk space")
    print(f"Directories processed: {removed + failed}")
    print(f"Directories {'' if not dry_run else 'that would be '}removed: {removed}")
    print(f"Failed removals: {failed}")
    
    return removed, failed, total_size

def main():
    parser = argparse.ArgumentParser(description="Clean up test database directories")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually remove anything, just show what would be removed")
    args = parser.parse_args()
    
    print(f"{'DRY RUN - ' if args.dry_run else ''}Searching for test database directories...")
    
    test_dirs = find_test_directories()
    if not test_dirs:
        print("No test directories found to clean up.")
        return
    
    print(f"Found {len(test_dirs)} test directories to clean up.")
    
    removed, failed, total_size = clean_test_directories(test_dirs, args.dry_run)
    
    if args.dry_run and removed > 0:
        print("\nTo actually remove these directories, run without --dry-run")

if __name__ == "__main__":
    main() 