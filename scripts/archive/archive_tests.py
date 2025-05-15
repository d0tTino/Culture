#!/usr/bin/env python
"""
Archive test data script for Culture.ai.
Archives test-related files to reduce the overall file count while preserving data.
"""

import os
import zipfile
import shutil
import re
from datetime import datetime
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("archive_tests")

# Test directories to archive
TEST_DIRS = [
    "tests",
    "benchmarks",
    "experiments"
]

# ChromaDB test directory patterns
CHROMADB_PATTERNS = [
    "chroma_db_test*",
    "test_chroma*",
    "chroma_benchmark*"
]

def archive_directory(dir_path, archive_name=None):
    """Archive a directory into a ZIP file."""
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        logger.warning(f"Directory {dir_path} does not exist, skipping")
        return False, 0
    
    # Count files before archiving
    file_count = sum(len(files) for _, _, files in os.walk(dir_path))
    
    if file_count == 0:
        logger.warning(f"Directory {dir_path} is empty, skipping")
        return False, 0
    
    if archive_name is None:
        archive_name = os.path.basename(dir_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"archives/{archive_name}_{timestamp}.zip"
    
    # Create archives directory if it doesn't exist
    os.makedirs("archives", exist_ok=True)
    
    try:
        logger.info(f"Archiving {dir_path} to {archive_path}")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=os.path.dirname(dir_path))
                    zipf.write(file_path, arcname)
        
        logger.info(f"Successfully archived {file_count} files from {dir_path}")
        return True, file_count
    except Exception as e:
        logger.error(f"Error archiving {dir_path}: {e}")
        return False, 0

def get_chroma_dirs():
    """Find all ChromaDB test directories."""
    chroma_dirs = []
    for pattern in CHROMADB_PATTERNS:
        chroma_dirs.extend(glob.glob(pattern))
    return chroma_dirs

def archive_chroma_dirs():
    """Archive all ChromaDB test directories."""
    chroma_dirs = get_chroma_dirs()
    
    if not chroma_dirs:
        logger.info("No ChromaDB test directories found.")
        return 0, 0
    
    logger.info(f"Found {len(chroma_dirs)} ChromaDB test directories to archive.")
    
    # Group them into a single archive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"archives/chromadb_test_dirs_{timestamp}.zip"
    
    # Create archives directory if it doesn't exist
    os.makedirs("archives", exist_ok=True)
    
    total_files = 0
    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for dir_path in chroma_dirs:
                if os.path.isdir(dir_path):
                    logger.info(f"Adding {dir_path} to archive...")
                    dir_files = 0
                    for root, _, files in os.walk(dir_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.join(dir_path, os.path.relpath(file_path, start=dir_path))
                            zipf.write(file_path, arcname)
                            dir_files += 1
                    logger.info(f"Added {dir_files} files from {dir_path}")
                    total_files += dir_files
        
        logger.info(f"Successfully archived {total_files} files from {len(chroma_dirs)} ChromaDB directories to {archive_path}")
        return len(chroma_dirs), total_files
    except Exception as e:
        logger.error(f"Error archiving ChromaDB directories: {e}")
        return 0, 0

def main():
    """Main function to archive test directories."""
    total_archived = 0
    total_files = 0
    
    # First archive regular test directories
    for dir_path in TEST_DIRS:
        success, file_count = archive_directory(dir_path)
        if success:
            total_archived += 1
            total_files += file_count
    
    # Next archive ChromaDB test directories
    chroma_dirs_count, chroma_files_count = archive_chroma_dirs()
    total_archived += chroma_dirs_count
    total_files += chroma_files_count
    
    logger.info(f"Archive process complete. Archived {total_files} files from {total_archived} directories.")
    
    if total_archived > 0:
        answer = input("Do you want to remove the original directories now that they are archived? (yes/no): ")
        if answer.lower() in ('yes', 'y'):
            # First remove regular test directories
            for dir_path in TEST_DIRS:
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        logger.info(f"Removed original directory: {dir_path}")
                    except Exception as e:
                        logger.error(f"Error removing directory {dir_path}: {e}")
            
            # Then remove ChromaDB test directories
            chroma_dirs = get_chroma_dirs()
            for dir_path in chroma_dirs:
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        logger.info(f"Removed ChromaDB directory: {dir_path}")
                    except Exception as e:
                        logger.error(f"Error removing ChromaDB directory {dir_path}: {e}")

if __name__ == "__main__":
    main() 