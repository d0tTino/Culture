import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_start_vllm_script(tmp_path: Path) -> None:
    pkg = tmp_path / "vllm" / "entrypoints" / "openai"
    pkg.mkdir(parents=True)
    # Ensure package structure
    (tmp_path / "vllm" / "__init__.py").write_text("")
    (tmp_path / "vllm" / "entrypoints" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")

    # Stub module that prints received arguments as JSON
    (pkg / "api_server.py").write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}:{env.get('PYTHONPATH', '')}"
    env["VLLM_PORT"] = "9999"
    env["VLLM_MODEL"] = "dummy/model"

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "start_vllm.sh"
    result = subprocess.run(
        ["bash", str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output == [
        "--model",
        "dummy/model",
        "--port",
        "9999",
        "--swap-space",
        "16",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "32",
        "--enable-chunked-prefill",
        "--download-dir",
        f"{Path.home()}/.cache/huggingface",
    ]
