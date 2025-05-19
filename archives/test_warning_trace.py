#!/usr/bin/env python
"""
Script to capture and print warning traces.
"""

import importlib.util
import os
import traceback
import warnings


def warning_handler(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that prints the full stack trace."""
    print("\n=== WARNING DETECTED ===")
    print(f"Warning: {message}")
    print(f"Category: {category.__name__}")
    print(f"Location: {filename}:{lineno}")

    if "pydantic" in str(message) and "Field" in str(message) and "required" in str(message):
        print("\n=== STACK TRACE ===")
        traceback.print_stack()

        print("\n=== MODULE INFO ===")
        module_name = os.path.basename(filename).replace(".py", "")
        if importlib.util.find_spec(module_name):
            module = importlib.import_module(module_name)
            print(f"Module: {module.__name__}")
            print(f"Package: {module.__package__}")
        else:
            print(f"Could not import module: {module_name}")

    return


# Install the custom warning handler
warnings.showwarning = warning_handler

print("Running pytest with custom warning handler...")
os.system("python -m pytest tests/unit/dspy/ -v")
