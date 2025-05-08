#!/usr/bin/env python
"""
Test script to debug logging functionality.
"""

import logging
import sys
import os

# Configure logging
LOG_FILE = "test_debug.log"

# Remove any existing log file
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)

# Set up a logger
logger = logging.getLogger("test_log_debug")

def main():
    """
    Test logging functionality.
    """
    logger.info("Starting log debug test")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test a custom logger
    custom_logger = logging.getLogger("custom_logger")
    custom_logger.info("This is a message from custom logger")
    
    logger.info("Log test complete")

if __name__ == "__main__":
    main() 