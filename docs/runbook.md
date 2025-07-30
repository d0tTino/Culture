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
   To use vLLM instead, first install the package and then follow the
   [Starting the vLLM Server](#starting-the-vllm-server) section below.
   ```bash
   pip install vllm  # install the vLLM server
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

## Starting the vLLM Server
Install vLLM if it is not already available:
```bash
pip install vllm
```
`scripts/start_vllm.sh` launches the vLLM OpenAI-compatible API. Set these
environment variables as needed:

- `VLLM_MODEL` – model name to load (defaults to `mistralai/Mistral-7B-Instruct-v0.2`)
- `VLLM_PORT` – server port (default `8001`)
- `VLLM_SWAP_SPACE` – swap space in GB (default `16`)

Example:

```bash
VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" VLLM_PORT=8001 scripts/start_vllm.sh
export VLLM_API_BASE="http://localhost:$VLLM_PORT"
export LLM_API_BASE="$VLLM_API_BASE"  # overrides Ollama when set
```

When `VLLM_API_BASE` (or `LLM_API_BASE` pointing to the same URL) is configured,
the application prefers the vLLM endpoint over Ollama. Unset this variable to
switch back to Ollama.

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

## Exporting Traces
Use `scripts/export_traces.py` to convert snapshots or event logs into a JSONL dataset.

```bash
# From stored snapshots
python scripts/export_traces.py --snapshots snapshots/ --output data/sample_traces.jsonl

# From a running Redpanda broker
ENABLE_REDPANDA=1 python scripts/export_traces.py --redpanda -o traces.jsonl
```

Each line in the output file is a JSON object representing a snapshot or event.

The script accepts optional filters:

- `--agent` – only include events with a matching `agent_id`
- `--start-step`/`--end-step` – restrict the step range

You can also generate a dataset with `make dataset`, overriding the input and
output paths if needed:

```bash
make dataset SNAPSHOTS=snapshots OUTPUT=data/traces.jsonl
```
The `make dataset` target uses this helper to export the most recent snapshots.

Run the simulation with `--export-dataset <file>` to automatically write a
dataset from the latest snapshots when the run finishes.

### Creating a JSONL Dataset
Generate a dataset from saved snapshots using `export_traces.py`. This example
writes the output to `data/traces.jsonl`:

```bash
python scripts/export_traces.py --snapshots snapshots/ --output data/traces.jsonl
```

Each line of `data/traces.jsonl` contains a single JSON object.

### Dataset Export
Follow these steps to manually export a dataset using `scripts/export_traces.py`:
1. Decide which source to use:
   - `--snapshots <DIR>` reads snapshots from a directory.
   - `--events <FILE>` loads a saved event log.
   - `--redpanda` pulls events from a running Redpanda broker.
2. Choose an output path with `-o`/`--output`.
3. Optionally filter the results with `--agent`, `--start-step`, or `--end-step`.
4. Run the script. For example:

   ```bash
   python scripts/export_traces.py --snapshots snapshots/ --output data/traces.jsonl
   ```
