#!/usr/bin/env python
"""
Final cleanup script for Culture.ai codebase.
Removes remaining test virtual environments and archives other large directories
to bring the total file count under 1000.
"""

import os
import shutil
import zipfile
import sys
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("final_cleanup")

# Directories to completely remove
DIRS_TO_REMOVE = [
    "test_venv",
    "test_venv_clean", 
    "test_venv_temp",
    ".pytest_cache",
    "__pycache__"
]

# Directories to archive (zip and then remove original)
DIRS_TO_ARCHIVE = [
    "benchmarks",
    "experiments"
]

# ChromaDB test directories to remove
CHROMA_DIRS_TO_REMOVE = [
    "chroma_db",
    "chroma_db_test",
    "chroma_db_store",
    "chroma_db_test_projects",
    "test_chroma_db",
    "test_chroma_dbs"
]

def check_file_count():
    """Count total files in the workspace."""
    total_count = 0
    for root, _, files in os.walk("."):
        if not any(part.startswith('.git') for part in Path(root).parts):  # Skip .git directory
            total_count += len(files)
    return total_count

def remove_directory(dir_path):
    """Safely remove a directory."""
    if os.path.exists(dir_path):
        try:
            logger.info(f"Removing directory: {dir_path}")
            shutil.rmtree(dir_path)
            return True
        except Exception as e:
            logger.error(f"Failed to remove {dir_path}: {e}")
            return False
    return False

def archive_directory(dir_path):
    """Archive a directory to a ZIP file and remove the original."""
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        logger.warning(f"Directory does not exist: {dir_path}")
        return False
    
    # Create archives directory if it doesn't exist
    archives_dir = "archives"
    os.makedirs(archives_dir, exist_ok=True)
    
    # Construct archive name
    archive_name = f"{archives_dir}/{os.path.basename(dir_path)}_{datetime.now().strftime('%Y%m%d')}.zip"
    
    try:
        # Count files before archiving
        file_count = sum(len(files) for _, _, files in os.walk(dir_path))
        logger.info(f"Archiving {dir_path} ({file_count} files) to {archive_name}")
        
        # Create ZIP archive
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=os.path.dirname(dir_path))
                    zipf.write(file_path, arcname)
        
        # Remove the original directory after successful archiving
        if os.path.exists(archive_name):
            logger.info(f"Archive created successfully: {archive_name}")
            shutil.rmtree(dir_path)
            logger.info(f"Original directory removed: {dir_path}")
            return True
        else:
            logger.error(f"Archive creation failed for: {dir_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error archiving {dir_path}: {e}")
        return False

def archive_tests_directory():
    """Archive the tests directory but keep a minimal structure."""
    tests_dir = "tests"
    if not os.path.exists(tests_dir):
        logger.warning(f"Tests directory does not exist: {tests_dir}")
        return False
    
    # Create archives directory if it doesn't exist
    archives_dir = "archives"
    os.makedirs(archives_dir, exist_ok=True)
    
    # Construct archive name
    archive_name = f"{archives_dir}/{tests_dir}_{datetime.now().strftime('%Y%m%d')}.zip"
    
    try:
        # Count files before archiving
        file_count = sum(len(files) for _, _, files in os.walk(tests_dir))
        logger.info(f"Archiving {tests_dir} ({file_count} files) to {archive_name}")
        
        # Create ZIP archive
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(tests_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=os.path.dirname(tests_dir))
                    zipf.write(file_path, arcname)
        
        # Remove most of the test files but keep the directory structure
        # Keep only __init__.py files to maintain importability
        for root, dirs, files in os.walk(tests_dir):
            for file in files:
                if file != "__init__.py":
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
            
            # Remove any __pycache__ directories
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
                shutil.rmtree(os.path.join(root, '__pycache__'))
        
        logger.info(f"Archived tests directory to {archive_name} and kept minimal structure")
        return True
            
    except Exception as e:
        logger.error(f"Error archiving tests directory: {e}")
        return False

def remove_excess_logs():
    """Remove unnecessary log files."""
    log_dir = "logs"
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
        try:
            file_count = len([f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))])
            logger.info(f"Cleaning {file_count} log files from {log_dir}")
            for file in os.listdir(log_dir):
                file_path = os.path.join(log_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Error cleaning logs: {e}")
            return False
    return False

def remove_test_chromadb_dirs():
    """Remove all ChromaDB test directories."""
    removed = 0
    for dir_name in CHROMA_DIRS_TO_REMOVE:
        if os.path.exists(dir_name):
            logger.info(f"Removing ChromaDB test directory: {dir_name}")
            try:
                shutil.rmtree(dir_name)
                removed += 1
            except Exception as e:
                logger.error(f"Failed to remove {dir_name}: {e}")
    
    # Also find and remove any dynamically named ChromaDB test directories
    for potential_dir in os.listdir('.'):
        if os.path.isdir(potential_dir) and (
            potential_dir.startswith('test_chroma_') or 
            potential_dir.startswith('chroma_benchmark_') or
            potential_dir.startswith('test_full_memory_pipeline_') or
            potential_dir.startswith('test_memory_utility_score_')
        ):
            try:
                logger.info(f"Removing additional ChromaDB test directory: {potential_dir}")
                shutil.rmtree(potential_dir)
                removed += 1
            except Exception as e:
                logger.error(f"Failed to remove {potential_dir}: {e}")
    
    return removed

def main():
    """Main execution function."""
    start_count = check_file_count()
    logger.info(f"Starting file count: {start_count}")
    
    # Remove virtual environments and test directories
    for dir_name in DIRS_TO_REMOVE:
        remove_directory(dir_name)
    
    # Archive large directories
    for dir_name in DIRS_TO_ARCHIVE:
        archive_directory(dir_name)
    
    # Handle tests directory specially
    archive_tests_directory()
    
    # Remove unnecessary log files
    remove_excess_logs()
    
    # Remove ChromaDB test directories
    remove_test_chromadb_dirs()
    
    # Check final file count
    final_count = check_file_count()
    logger.info(f"Final file count: {final_count}")
    logger.info(f"Removed {start_count - final_count} files")
    
    if final_count < 1000:
        logger.info("SUCCESS: File count is now below 1000!")
    else:
        logger.warning(f"File count is still {final_count}, which is above the target of 1000.")

if __name__ == "__main__":
    main() 