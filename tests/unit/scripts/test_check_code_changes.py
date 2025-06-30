import subprocess
from pathlib import Path

import pytest


@pytest.mark.unit
def test_docstring_only_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / ".gitattributes").write_text("* text=auto eol=lf")

    file_path = repo / "foo.py"
    file_path.write_text('def func():\n    """Original docstring."""\n    pass\n')
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    file_path.write_text('def func():\n    """Updated docstring."""\n    pass\n')
    subprocess.run(["git", "add", "foo.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "doc change"], cwd=repo, check=True)

    diff_output = subprocess.run(
        ["git", "diff", base_sha, "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    added_lines = [line[1:].strip() for line in diff_output.splitlines() if line.startswith("+")]
    # Filter out the +++ line
    added_lines = [line for line in added_lines if not line.startswith("++")]
    assert len(added_lines) == 1
    assert added_lines[0] == '"""Updated docstring."""'
