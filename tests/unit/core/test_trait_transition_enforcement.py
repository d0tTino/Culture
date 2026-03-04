from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
ENGINE_FILE = SRC_ROOT / "agents" / "core" / "personality_engine.py"


@pytest.mark.unit
def test_no_legacy_trait_drift_helpers_are_used_in_src() -> None:
    for path in SRC_ROOT.rglob("*.py"):
        source = path.read_text()
        assert "apply_trait_drift(" not in source
        assert "update_traits(" not in source


@pytest.mark.unit
def test_only_personality_engine_mutates_state_traits() -> None:
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == "state"
                    and target.value.attr == "traits"
                    and path != ENGINE_FILE
                ):
                    raise AssertionError(f"Illegal trait write outside personality engine: {path}")
