import os
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

    file_path = repo / "foo.py"
    file_path.write_text('def func():\n    """Original docstring."""\n    pass\n')
    subprocess.run(["git", "add", "foo.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)

    file_path.write_text('def func():\n    """Updated docstring."""\n    pass\n')
    subprocess.run(["git", "add", "foo.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "doc change"], cwd=repo, check=True)

    output_file = repo / "out.txt"
    env = os.environ.copy()
    env["GITHUB_OUTPUT"] = str(output_file)
    script = Path(__file__).resolve().parents[3] / "scripts" / "check_code_changes.sh"
    subprocess.run(["bash", str(script)], cwd=repo, env=env, check=True)

    assert "CODE_CHANGES=false" in output_file.read_text()
