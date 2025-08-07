import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_quickstart_script_smoke() -> None:
    script_path = Path(__file__).resolve().parents[1] / ".." / "scripts" / "quickstart.sh"
    script_path = script_path.resolve()
    result = subprocess.run(
        ["bash", str(script_path), "--smoke-test"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    output = result.stdout.splitlines()
    assert "Installing dependencies..." in output
    assert "Launching vLLM..." in output
    assert "Starting simulation..." in output
    assert "Joining Discord channel..." in output
