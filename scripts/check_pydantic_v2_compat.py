#!/usr/bin/env python3
"""Check for Pydantic v1 API usage.

This script scans tracked Python files and reports deprecated patterns
that are incompatible with Pydantic v2.
"""
import ast
import subprocess
import sys
from pathlib import Path

DEPRECATED_ATTRS = {
    "dict": "model_dump",
    "json": "model_dump_json",
    "schema": "model_json_schema",
}
DEPRECATED_DECORATORS = {
    "validator": "field_validator",
    "root_validator": "model_validator",
}


def tracked_python_files() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "*.py"], capture_output=True, text=True, check=True
    )
    return [Path(p) for p in proc.stdout.splitlines()]


class Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[tuple[int, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for item in node.body:
            if isinstance(item, ast.ClassDef) and item.name == "Config":
                self.issues.append(
                    (
                        item.lineno,
                        "Pydantic 'Config' class found; use 'model_config' dict",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in DEPRECATED_ATTRS:
            replacement = DEPRECATED_ATTRS[node.func.attr]
            self.issues.append(
                (node.lineno, f"Deprecated '{node.func.attr}' method; use '{replacement}'")
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id in DEPRECATED_DECORATORS:
                replacement = DEPRECATED_DECORATORS[dec.id]
                self.issues.append(
                    (dec.lineno, f"Deprecated '@{dec.id}' decorator; use '@{replacement}'")
                )
        self.generic_visit(node)


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - parse errors
        return [f"Failed to parse {path}: {exc}"]
    visitor = Visitor()
    visitor.visit(tree)
    return [f"{path}:{line}: {msg}" for line, msg in visitor.issues]


def main() -> int:
    has_errors = False
    for file in tracked_python_files():
        for issue in check_file(file):
            print(issue, file=sys.stderr)
            has_errors = True
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
