import re
import subprocess
import sys

FORBIDDEN_PATTERN = re.compile(r"^\s*(from\s+dspy\b|import\s+dspy(\s|$))")
ALLOWED_FILES = {"src/dspy_ai/__init__.py"}


def main() -> None:
    """
    Check for direct imports of the `dspy` package in Python files under `src/`.
    Exits with a non-zero status code if forbidden imports are found.
    """
    try:
        # Get all tracked Python files in the src directory
        git_files_proc = subprocess.run(
            ["git", "ls-files", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        all_py_files = git_files_proc.stdout.splitlines()
        src_py_files = [
            f for f in all_py_files if f.startswith("src/") and f not in ALLOWED_FILES
        ]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error getting files from git: {e}", file=sys.stderr)
        sys.exit(1)

    found_forbidden_import = False
    for filepath in src_py_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if FORBIDDEN_PATTERN.search(line):
                        print(
                            f"{filepath}:{i}: Found forbidden direct 'dspy' import.",
                            file=sys.stderr,
                        )
                        found_forbidden_import = True
        except FileNotFoundError:
            # This can happen if a file was deleted but is still in the index
            continue
        except Exception as e:
            print(f"Error reading file {filepath}: {e}", file=sys.stderr)
            continue

    if found_forbidden_import:
        print(
            "\nDirect 'dspy' imports are forbidden in the 'src/' directory.",
            file=sys.stderr,
        )
        print("Please use 'from src import dspy_ai as dspy' instead.", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main() 