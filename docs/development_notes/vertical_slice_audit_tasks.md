# Vertical Slice Audit: Task List

This task list distills the key actions from the May 2025 audit report. Completing these items will ensure a smooth plug-and-play workflow for the Windows vertical slice and lay the groundwork for future Discord orchestration.

## Dependency and Build

- [x] Freeze all core library versions in `requirements.txt` using `pip-compile`.
- [x] Pin the CUDA build of PyTorch (`torch==2.3.0+cu121`).
- [x] Pin `ollama>=0.1.34` and `chromadb==0.4.24`.

## OS-Specific Fixes

- [x] Provide a Windows fallback for `uvloop` and normalise paths with `pathlib`.
- [x] Update documentation to include Windows venv activation commands.

## Observability

- [x] Enable the OTEL exporter in `infra/logging_config.py`.
- [x] Add instructions to set `DEBUG_SQLITE=1` for debugging database locks.

## Continuous Integration

- [x] Add Ruff and mypy strict checks to pre-commit and CI workflows.
- [x] Add a `windows-latest` job to the GitHub Actions test matrix.
- [x] Introduce a red-team gate using Garak/jailbreak corpora.

## Deterministic Replay

- [x] Implement checkpoint and replay support with a minimal `redpanda` log.
- [x] Save `collective_metrics`, `knowledge_board`, and agent states every 100 ticks.

## Next Sprint Prep

- [x] Support multiple Discord bot tokens via separate gateway shards.
- [ ] Map agent IDs to bot tokens in a shared PostgreSQL table.
- [x] Filter outgoing messages through an OPA policy engine.

Completing these tasks will bring the project in line with the audit's recommendations and ready the codebase for stable multi-bot orchestration.
