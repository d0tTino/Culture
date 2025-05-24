#!/usr/bin/env python3
"""
Script to fix import paths in all test files after reorganization.

This script updates sys.path.append lines in test files to correctly point
to the project root directory.
"""

import os
import re


def fix_imports_in_file(file_path: str) -> None:
    """
    Fix the import paths in a single test file.

    Args:
        file_path: Path to the test file to fix

    Returns:
        bool: True if the file was modified, False otherwise
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Look for sys.path.append patterns that need fixing
    old_pattern = r"sys\.path\.append\(str\(Path\(__file__\)\.parent(\)?)\)"
    new_path = r"sys.path.append(str(Path(__file__).parent.parent.parent))"

    # Check if we need to replace anything
    if not re.search(old_pattern, content):
        return

    # Replace the pattern
    new_content = re.sub(old_pattern, new_path, content)

    # Write the modified content back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return


def main() -> None:
    """
    Find all Python test files in the tests directory and fix their imports.
    """
    # Get all Python files in the tests directory
    tests_dir = os.path.join("tests", "integration")
    modified_files = 0

    for filename in os.listdir(tests_dir):
        if filename.endswith(".py"):
            file_path = os.path.join(tests_dir, filename)
            fix_imports_in_file(file_path)
            print(f"Checked imports in {file_path}")
            modified_files += 1

    print(f"\nChecked imports in {modified_files} files.")


if __name__ == "__main__":
    main()
