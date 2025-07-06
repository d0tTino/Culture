# Runbook

This runbook outlines routine operations for working with Culture.ai.

## Starting a Simulation
1. Activate your virtual environment (on Windows run `venv\Scripts\activate.bat` or `.venv\Scripts\activate.bat`).
2. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
3. Pull the required model and start Ollama:
   ```bash
   ollama pull mistral:latest
   ```
   Alternatively, start a vLLM server with swap space enabled to avoid
   out-of-memory errors when running many agents. The helper script
   `start_vllm.sh` launches the server and you can point the application to it:
   ```bash
   # Optionally override the model or port used by vLLM (defaults to port 8001)
   VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" VLLM_PORT=8001 \
   scripts/start_vllm.sh
   export VLLM_API_BASE="http://localhost:$VLLM_PORT"
   # When set, the application uses the vLLM OpenAI-compatible endpoint
   ```
4. (Optional) Start the vector store:
   ```bash
   docker compose up -d
   ```
5. Run the simulation:
   ```bash
   python -m src.app --steps 5
   ```
6. (Optional) Save or resume using a checkpoint:
   ```bash
   python -m src.app --steps 5 --checkpoint my_sim.pkl
   ```
7. (Optional) Replay a previous run deterministically:
   ```bash
   python -m src.app --steps 5 --checkpoint my_sim.pkl --replay
   ```
   When `--replay` is provided and `ENABLE_REDPANDA=1`, the simulation will
   restore RNG/environment state and replay agent actions from the Redpanda
   event log.
8. Snapshots of the simulation state are written every 100 ticks. Set the
   `SNAPSHOT_COMPRESS` environment variable to `1` to save them as
   zstandard-compressed files (`snapshot_<step>.json.zst`). Use the `zstd`
   command line tool or Python's `zstandard` module to decompress them, e.g.:

   ```bash
   zstd -d snapshot_100.json.zst -o snapshot_100.json
   ```

## Running Tests
Run the full suite with coverage:
```bash
python -m pytest --cov=src --cov-report=term-missing tests/
```

## Troubleshooting
- **Missing environment variables**: copy `.env.example` to `.env` and update values.
- **Vector store errors**: ensure `docker compose up -d` is running or switch to ChromaDB.
- **LLM timeouts**: check `OLLAMA_API_BASE` and network connectivity.

See the [Quickstart for Developers](../README.md#quickstart-for-developers) for additional setup details.
