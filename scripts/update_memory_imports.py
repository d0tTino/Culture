#!/usr/bin/env python
"""
Script to update import statements for the memory reorganization.

This script finds all Python files in the project and updates import statements
from src.infra.memory.vector_store to src.agents.memory.vector_store.
"""

import re
from pathlib import Path
from typing import List, Tuple


def find_python_files(base_dir: str = ".") -> List[Path]:
    """Find all Python files in the project."""
    base_path = Path(base_dir)
    python_files = list(base_path.glob("**/*.py"))

    # Filter out some specific directories we don't want to modify
    return [f for f in python_files if not any(part.startswith(".") for part in f.parts)]


def update_imports_in_file(file_path: Path) -> Tuple[int, List[str]]:
    """
    Update import statements in a single file.

    Returns:
        Tuple of (number of changes, list of changed lines)
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Define patterns to search for
    patterns = [
        (
            r"from src\.infra\.memory\.vector_store import",
            "from src.agents.memory.vector_store import",
        ),
        (r"import src\.infra\.memory\.vector_store", "import src.agents.memory.vector_store"),
        # Add more patterns if needed
    ]

    changes = []
    change_count = 0

    # Apply each pattern
    for pattern, replacement in patterns:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            change_count += len(matches)
            changes.extend(matches)

    # Write back to the file if changes were made
    if change_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    return change_count, changes


def main():
    """Main function to update imports across the codebase."""
    print("Updating memory import statements across the codebase...")

    # Find all Python files
    python_files = find_python_files()
    print(f"Found {len(python_files)} Python files to process")

    # Track changes
    total_changes = 0
    files_changed = 0

    # Process each file
    for file_path in python_files:
        changes, changed_lines = update_imports_in_file(file_path)

        if changes > 0:
            files_changed += 1
            total_changes += changes
            print(f"Updated {changes} imports in {file_path}")
            for line in changed_lines:
                print(f"  - {line}")

    print(f"\nSummary: Updated {total_changes} import statements in {files_changed} files")


if __name__ == "__main__":
    main()
