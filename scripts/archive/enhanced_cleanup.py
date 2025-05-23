#!/usr/bin/env python
"""
Enhanced cleanup script for Culture.ai codebase.
This script aggressively removes test data, cache files, and other unnecessary files
to bring the codebase under 1000 files.
"""

import os
import shutil
import re
import argparse
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_cleanup")

# Regular expressions to match test database directories
TEST_DIR_PATTERNS = [
    r'test_chroma_.*',
    r'test_.*_dbs',
    r'test_full_memory_pipeline_.*',
    r'test_memory_utility_score_.*',
    r'chroma_benchmark_.*'
]

# Patterns for files that can be safely deleted
SAFE_DELETE_FILE_PATTERNS = [
    r'.*\.pyc$',
    r'.*\.pyo$',
    r'.*\.pyd$',
    r'.*\.log$',
    r'.*\.egg-info$',
    r'.*\.coverage$',
    r'.*\.pytest_cache$',
    r'.*\.ipynb_checkpoints$'
]

# Directories to always exclude from cleanup to preserve essential code
EXCLUDE_DIRS = [
    '.git',
    'src',
    'docs'
]

def find_test_directories(base_dir='.'):
    """Find all test database directories that match our patterns."""
    test_dirs = []
    
    for root, dirs, _ in os.walk(base_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            
            # Skip directories we've explicitly excluded
            if any(excluded in full_path for excluded in EXCLUDE_DIRS):
                continue
                
            # Skip hidden directories
            if dir_name.startswith('.'):
                continue
                
            # Check if the directory name matches any of our patterns
            for pattern in TEST_DIR_PATTERNS:
                if re.fullmatch(pattern, dir_name) or re.fullmatch(pattern, full_path[2:]):
                    test_dirs.append(full_path)
                    break
    
    return test_dirs

def find_pycache_dirs(base_dir='.'):
    """Find all __pycache__ directories."""
    pycache_dirs = []
    
    for root, dirs, _ in os.walk(base_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_dirs.append(os.path.join(root, dir_name))
    
    return pycache_dirs

def find_venv_dirs(base_dir='.'):
    """Find virtual environment directories."""
    venv_dirs = []
    
    for root, dirs, _ in os.walk(base_dir):
        # Only look at top level directories
        if root != base_dir:
            continue
            
        for dir_name in dirs:
            if dir_name.startswith('venv') or dir_name.startswith('test_venv') or 'env' in dir_name.lower():
                # Keep one venv directory for development (test_venv_clean)
                if dir_name == 'test_venv_clean':
                    continue
                venv_dirs.append(os.path.join(root, dir_name))
    
    return venv_dirs

def find_unnecessary_files(base_dir='.'):
    """Find files that match our safe-to-delete patterns."""
    unnecessary_files = []
    
    for root, _, files in os.walk(base_dir):
        # Skip excluded directories
        if any(excluded in root for excluded in EXCLUDE_DIRS):
            continue
            
        for file_name in files:
            full_path = os.path.join(root, file_name)
            
            # Check if the file matches any of our patterns
            for pattern in SAFE_DELETE_FILE_PATTERNS:
                if re.match(pattern, file_name):
                    unnecessary_files.append(full_path)
                    break
    
    return unnecessary_files

def check_directory_size(dir_path):
    """Calculate size of a directory in MB."""
    try:
        total_size = 0
        for dirpath, _, filenames in os.walk(dir_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
        
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception as e:
        logger.error(f"Error calculating size of {dir_path}: {e}")
        return 0

def remove_items(items, item_type="directory", dry_run=True):
    """Remove the specified items (directories or files)."""
    removed = 0
    failed = 0
    total_size = 0
    
    for item_path in items:
        try:
            # Calculate size
            if os.path.isdir(item_path):
                item_size = check_directory_size(item_path)
            else:
                item_size = os.path.getsize(item_path) / (1024 * 1024)  # Convert to MB
            
            total_size += item_size
            
            if dry_run:
                logger.info(f"Would remove {item_type}: {item_path} ({item_size:.2f} MB)")
            else:
                logger.info(f"Removing {item_type}: {item_path} ({item_size:.2f} MB)")
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            removed += 1
        except Exception as e:
            logger.error(f"Error removing {item_path}: {e}")
            failed += 1
    
    action = "Would free" if dry_run else "Freed"
    logger.info(f"\n{action} {total_size:.2f} MB of disk space")
    logger.info(f"{item_type.capitalize()}s processed: {removed + failed}")
    logger.info(f"{item_type.capitalize()}s {'' if not dry_run else 'that would be '}removed: {removed}")
    logger.info(f"Failed removals: {failed}")
    
    return removed, failed, total_size

def check_file_count():
    """Count total files in the project (excluding .git)."""
    total = 0
    for root, _, files in os.walk('.'):
        if '.git' in root:
            continue
        total += len(files)
    return total

def main():
    parser = argparse.ArgumentParser(description="Aggressively clean up test and unnecessary files")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually remove anything, just show what would be removed")
    parser.add_argument("--keep-venv", action="store_true", help="Don't remove virtual environment directories")
    args = parser.parse_args()
    
    logger.info(f"{'DRY RUN - ' if args.dry_run else ''}Starting enhanced cleanup...")
    
    initial_count = check_file_count()
    logger.info(f"Initial file count: {initial_count}")
    
    # 1. Find and remove test directories
    logger.info("Finding test database directories...")
    test_dirs = find_test_directories()
    if test_dirs:
        logger.info(f"Found {len(test_dirs)} test directories to clean up.")
        remove_items(test_dirs, "test directory", args.dry_run)
    else:
        logger.info("No test directories found to clean up.")
    
    # 2. Find and remove __pycache__ directories
    logger.info("Finding __pycache__ directories...")
    pycache_dirs = find_pycache_dirs()
    if pycache_dirs:
        logger.info(f"Found {len(pycache_dirs)} __pycache__ directories to clean up.")
        remove_items(pycache_dirs, "__pycache__ directory", args.dry_run)
    else:
        logger.info("No __pycache__ directories found to clean up.")
    
    # 3. Find and remove virtual environments if not explicitly kept
    if not args.keep_venv:
        logger.info("Finding virtual environment directories...")
        venv_dirs = find_venv_dirs()
        if venv_dirs:
            logger.info(f"Found {len(venv_dirs)} virtual environment directories to clean up.")
            remove_items(venv_dirs, "virtual environment directory", args.dry_run)
        else:
            logger.info("No virtual environment directories found to clean up.")
    
    # 4. Find and remove unnecessary files
    logger.info("Finding unnecessary files (pyc, pyo, logs, etc.)...")
    unnecessary_files = find_unnecessary_files()
    if unnecessary_files:
        logger.info(f"Found {len(unnecessary_files)} unnecessary files to clean up.")
        remove_items(unnecessary_files, "file", args.dry_run)
    else:
        logger.info("No unnecessary files found to clean up.")
    
    if not args.dry_run:
        final_count = check_file_count()
        logger.info(f"Final file count: {final_count}")
        logger.info(f"Removed {initial_count - final_count} files")
        
        if final_count > 1000:
            logger.warning(f"Still have {final_count} files (target: <1000). Consider additional cleanup measures.")
        else:
            logger.info("SUCCESS! File count is now below 1000.")
    
    if args.dry_run:
        logger.info("\nTo actually remove these items, run without --dry-run")

if __name__ == "__main__":
    main() 