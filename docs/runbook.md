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
   out-of-memory errors when running many agents:
   ```bash
   scripts/start_vllm.sh  # defaults to port 8001 (override with VLLM_PORT)
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
