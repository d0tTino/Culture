#!/usr/bin/env python3
"""
Script to fix import paths in all test files after reorganization.

This script updates sys.path.append lines in test files to correctly point
to the project root directory.
"""

import re
from pathlib import Path


def fix_imports_in_file(file_path: Path) -> bool:
    """
    Fix the import paths in a single test file.

    Args:
        file_path: Path to the test file to fix.

    Returns:
        bool: True if the file was modified, False otherwise.
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Look for sys.path.append patterns that need fixing
    old_pattern = r"sys\.path\.append\(str\(Path\(__file__\)\.parent(\)?)\)"
    new_path = r"sys.path.append(str(Path(__file__).parent.parent.parent))"

    # Check if we need to replace anything
    if not re.search(old_pattern, content):
        return False

    # Replace the pattern
    new_content = re.sub(old_pattern, new_path, content)

    # Write the modified content back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return True


def main() -> None:
    """
    Find all Python test files in the tests directory and fix their imports.
    """
    # Get all Python files in the tests directory
    tests_dir = Path("tests") / "integration"
    modified_files = 0

    for file_path in tests_dir.iterdir():
        if file_path.suffix == ".py":
            if fix_imports_in_file(file_path):
                modified_files += 1
                print(f"Fixed imports in {file_path}")
            else:
                print(f"Checked imports in {file_path}")

    print(f"\nChecked imports in {modified_files} files.")


if __name__ == "__main__":
    main()
