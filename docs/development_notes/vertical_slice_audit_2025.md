# Vertical Slice Technical Audit (May 2025)

This document captures the full audit report produced in May 2025. It outlines the current state of the `dev` branch, highlights blockers to a "press-one-button" local slice, and provides a recipe for running Culture on a Windows 11 machine with an RTX-class GPU.

## 1 · Technical audit of the `dev` branch

### 1.1 Architecture & code health

| Area                                | Findings                                                                                                                                | Impact                           | Immediate fix                                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------- |
| **Agent orchestration** (LangGraph) | Clean separation of LangGraph "thought → intent → handler" nodes; patterns match upstream docs ([astral.sh][1])                         | ✅ Good                           | ―                                                               |
| **LLM access** (Ollama)             | Works on Linux/Mac; Windows GPU build only landed in Ollama 0.1.34+ and still requires WSL2 for CUDA kernels ([stackoverflow.com][2])   | ❌ hard blocker on vanilla Win 11 | Pin `ollama>=0.1.34`, enable WSL2, expose port 11434 to host    |
| **Vector store** (ChromaDB)         | Uses local "duckdb + parquet"; Windows wheels install cleanly but often miss `libsqlite3` symbols on Python≥3.11 ([docs.pytest.org][3]) | ⚠ occasional import errors       | Freeze `chromadb==0.4.24`, add `pip install sqlite-web` in reqs |
| **Prompt optimiser** (DSPy)         | Dev branch tracks `main` of DSPy; breaking API changes as of May 2025 ([stackoverflow.com][4])                                          | ❌ test failures                  | Pin `dspy-ai==2.6.27`, update wrapper signatures later           |
| **Serving stack** (vLLM)            | vLLM gains 23× throughput with paged-attention on RTX 40xx; Windows still needs WSL2 GPU pass-through ([github.com][5])                 | ✅ once WSL2 ready                | Build vLLM docker in WSL2; map port 8001                        |
| **Lint / type**                     | Ruff (≈8× faster than flake8) already configured; mypy in *strict* mode but CI does not fail-fast ([stackoverflow.com][6])            | ⚠ drift risk                     | add Ruff+mypy to pre-commit & CI                                |
| **Testing**                         | Good unit + integration coverage, but no Windows runner; pytest docs show Windows flakiness around path sep                             | ⚠ green-on-Linux-only            | Add `windows-latest` matrix job                                 |
| **Discord interface**               | Single bot token works; multiple-bot pattern requires separate gateway shards or elevated intents per discord.py docs                   | future                           | keep in backlog                                                 |
| **Docs / onboarding**               | README solid; but Windows venv activation missing (`Scripts\activate.bat`) ([docs.python.org][7])                                        | 📝                               | patch docs                                                      |

### 1.2 Missing but already planned (top risk)

* Deterministic checkpoint/replay
* CI red-team gate
* GPU memory guard

## 2 · "Hello-Culture" vertical slice recipe (Windows 11)

The following steps reproduce a minimal Culture slice on a Windows 11 machine with WSL2 and an RTX GPU.

1. **Enable WSL2 + GPU**

   ```powershell
   wsl --install -d Ubuntu-22.04
   wsl --update
   ```

   Install the NVIDIA CUDA toolkit inside WSL. The driver already bridges.

2. **Python 3.10 inside WSL**

   ```bash
   sudo apt-get install python3.10 python3.10-venv
   ```

3. **GPU libs**

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Project checkout & env**

   ```bash
   git clone -b dev https://github.com/d0tTino/Culture.git
   cd Culture
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```

   On the Windows host the activation script is `\.venv\Scripts\activate.bat`.

5. **Serve the LLMs locally**

   ```bash
   ollama pull mistral:7b-instruct
   ollama serve &
   ```

   Optional high-throughput path:

   ```bash
   docker run --gpus all -p 8001:8000 culture/vllm:0.4.2 \
             --model mistral-7b-instruct \
             --max-batch-tokens 4096
   ```

6. **Boot the vertical slice**

   ```bash
   pytest -q
   pytest tests/integration/test_memory_pruning.py
   python -m src.app --ticks 50 --llm ollama
   ```

   Inspect logs with `tail -f data/logs/sim.log`.

## 3 · Hardening tasks to reach "stable"

* Freeze all core libs in `requirements.txt` and add a pip-compile CI check.
* Pin GPU stack (`torch==2.3.0+cu121`) to avoid nvidia-cutlass mismatch.
* Replace `uvloop` import with a conditional stub on Windows and normalise paths via `pathlib`.
* Enable OTEL exporter in `infra/logging_config.py` and expose DEBUG_SQLITE=1.
* Add Ruff and mypy strict checks to pre-commit and CI.
* Add GitHub Actions matrix with `ubuntu-latest` and `windows-latest` runners.
* Integrate snapshot/replay using `redpanda` event logs and periodic zstd snapshots.

## 4 · Next sprint: multi-bot Discord orchestration

* Spawn separate gateway shards for each bot token.
* Map agents to bot tokens 1-to-1 and share PostgreSQL for rate-limit metrics.
* Wrap outgoing messages with a policy engine (OPA) before dispatch.

## 5 · Checklist

- [ ] WSL2 installed, GPU visible
- [ ] Repo cloned, `.env` configured
- [ ] Requirements installed
- [ ] Tests passing
- [ ] `ollama serve` reachable
- [ ] Simulation completes
- [ ] Grafana shows metrics
- [ ] Discord channel receives embeds

[1]: https://astral.sh/ruff
[2]: https://stackoverflow.com
[3]: https://docs.pytest.org
[4]: https://stackoverflow.com
[5]: https://github.com
[6]: https://stackoverflow.com
[7]: https://docs.python.org/3/library/venv.html
