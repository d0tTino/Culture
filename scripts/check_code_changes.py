import os
import sys
from pathlib import Path

try:
    import git
except ImportError:
    print("GitPython is not installed. Please install it with 'pip install GitPython'")
    sys.exit(1)


def is_code_file(file_path: str) -> bool:
    """Check if a file is considered a code file (not documentation)."""
    return not (file_path.endswith(".md") or file_path.startswith("docs/"))


def has_code_changes(diff: git.Diff) -> bool:
    """
    Analyzes a diff to determine if it contains meaningful code changes.
    Returns True if changes are outside of comments and docstrings.
    """
    if not diff.diff:
        return False

    # The diff content is already a string, no need to decode
    diff_text = diff.diff
    in_multiline_docstring = False
    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue

        if '"""' in line or "'''" in line:
            in_multiline_docstring = not in_multiline_docstring
            continue

        if in_multiline_docstring:
            continue

        if line.startswith("+"):
            stripped_line = line[1:].strip()
            if stripped_line and not stripped_line.startswith(("#", "//", "*")):
                return True  # Found a non-comment, non-docstring addition

    return False


def main() -> None:
    repo_path = Path.cwd()
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        print(f"Error: Not a git repository: {repo_path}", file=sys.stderr)
        sys.exit(1)

    # Determine base commit for diff
    if "GITHUB_BASE_SHA" in os.environ:
        base_commit_sha = os.environ["GITHUB_BASE_SHA"]
    elif "GITHUB_EVENT_BEFORE" in os.environ:
        base_commit_sha = os.environ["GITHUB_EVENT_BEFORE"]
    else:
        # Fallback for local execution: compare against the previous commit
        if len(repo.heads.main.log()) > 1:
            base_commit_sha = "HEAD~1"
        else:  # Handle initial commit
            base_commit_sha = repo.head.commit.hexsha

    try:
        base_commit = repo.commit(base_commit_sha)
        head_commit = repo.head.commit
    except git.BadName as e:
        print(f"Error finding commit: {e}", file=sys.stderr)
        # If we can't find the base commit, assume there are changes
        code_changes_found = True
        base_commit = head_commit = repo.head.commit  # dummy

    diff_index = base_commit.diff(head_commit)

    code_changes_found = False
    for diff in diff_index:
        if diff.change_type in ("D", "R"):  # Deletion or Rename
            code_changes_found = True
            break
        if not diff.a_path or not is_code_file(diff.a_path):
            continue
        if diff.b_blob is None:  # Deleted file
            continue

        if has_code_changes(diff):
            code_changes_found = True
            break

    output_file_path = os.getenv("GITHUB_OUTPUT")
    output = f"CODE_CHANGES={str(code_changes_found).lower()}"

    if output_file_path:
        with open(output_file_path, "a") as f:
            f.write(f"{output}\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
