#!/usr/bin/env python
"""
Run script for testing level 2 hierarchical memory consolidation.
This script will execute the test_level2_memory_consolidation test
and output results to the console and a log file.
"""

import logging
import sys
import time
from test_level2_memory_consolidation import test_level2_memory_consolidation

if __name__ == "__main__":
    print("Starting level 2 memory consolidation test...")
    start_time = time.time()
    
    # Run the test
    test_level2_memory_consolidation()
    
    elapsed_time = time.time() - start_time
    print(f"Test completed in {elapsed_time:.2f} seconds")
    print("Check level2_memory_consolidation_test.log for detailed results") 