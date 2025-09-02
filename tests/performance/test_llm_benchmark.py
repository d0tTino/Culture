import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_llm", Path(__file__).parents[2] / "scripts" / "benchmark_llm.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.performance
def test_llm_benchmark(tmp_path: Path) -> None:
    mod = _load_script()
    output = tmp_path / "results.json"
    with (
        patch.object(mod, "time_calls", side_effect=[[0.1], [0.2]]),
        patch.object(mod.infra_config, "load_config", return_value={}),
        patch.object(mod.llm_client, "get_llm_client"),
        patch.object(mod.llm_client, "generate_text", return_value=None),
    ):
        argv = [
            "benchmark_llm.py",
            "hi",
            "--runs",
            "1",
            "--model",
            "dummy",
            "--vllm_base",
            "http://fake",
            "--output",
            str(output),
        ]
        with patch.object(sys, "argv", argv):
            mod.main()
    results = json.loads(output.read_text())
    assert results["vllm"]["avg_latency"] < results["ollama"]["avg_latency"]
