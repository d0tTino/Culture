# Runbook

This runbook outlines routine operations for working with Culture.ai.

## Starting a Simulation
1. Activate your virtual environment (on Windows run `venv\Scripts\activate.bat` or `.venv\Scripts\activate.bat`).
2. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
3. Start an LLM server (vLLM is preferred):
   ```bash
   pip install vllm  # install the vLLM server
   VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" VLLM_PORT=8001 scripts/start_vllm.sh
   export VLLM_API_BASE="http://localhost:$VLLM_PORT"
   ```
   `scripts/start_vllm.sh` enables continuous batching via
   `--enable-chunked-prefill` and resolves models from the local Hugging Face
   cache first (override with `VLLM_DOWNLOAD_DIR`). Adjust
   `VLLM_MAX_BATCH_TOKENS` and `VLLM_MAX_NUM_SEQS` to control batch size.
   Smaller, colon-delimited models (for example `mistral:latest`) can still run
   via Ollama, which avoids GPU overhead for lightweight models. Ollama can be
   used as a fallback backend:
   ```bash
   ollama pull mistral:latest
   ollama serve &
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

## Start vLLM
Install vLLM if it is not already available:
```bash
pip install vllm
```
`scripts/start_vllm.sh` launches the vLLM OpenAI-compatible API. Set these
environment variables before running it:

- `VLLM_MODEL` – model name to load (defaults to `mistralai/Mistral-7B-Instruct-v0.2`)
- `VLLM_PORT` – server port (default `8001`)
- `VLLM_SWAP_SPACE` – swap space in GB (default `16`)

Example:

```bash
VLLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" VLLM_PORT=8001 scripts/start_vllm.sh
export VLLM_API_BASE="http://localhost:$VLLM_PORT"
export LLM_API_BASE="$VLLM_API_BASE"  # overrides Ollama when set
```

Run `scripts/benchmark_llm.py` after the server starts to compare vLLM and Ollama performance. The script reports average latency, throughput, and an approximate cost per million tokens:

| Backend | Avg latency (s) | Throughput (req/s) | Cost per 1M tokens* |
|---------|----------------:|-------------------:|--------------------:|
| vLLM    | 0.25            | 4.00               | ~$0 (local GPU)     |
| Ollama  | 1.20            | 0.83               | ~$0 (local CPU)     |

*Hardware and electricity costs not included; remote APIs would add per-token charges.

Add `--output results.json` to persist the results for later analysis.

Results will vary depending on hardware and models.

When `VLLM_API_BASE` (or `LLM_API_BASE` pointing to the same URL) is configured, the application prefers the vLLM endpoint over Ollama. Unset this variable to switch back to Ollama.

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

## Public Persistent World Profile
Use the `public_persistent` profile for long-lived public deployments:

```bash
PROFILE=public_persistent scripts/start_public_persistent.sh --steps 100
```

This launcher validates Python/runtime dependencies and checks LLM connectivity before booting.

## Incident Recovery
When production traffic degrades or halts:
1. Check subsystem status and readiness:
   ```bash
   curl -s http://localhost:8000/health | jq
   curl -s -o /tmp/ready.json -w "%{http_code}\n" http://localhost:8000/ready
   cat /tmp/ready.json | jq
   ```
2. If `llm` is degraded, restore backend first (`scripts/start_vllm.sh` or Ollama fallback) and retest `/ready`.
3. If `event_bus` is degraded, restart the app process to reinitialize subscribers.
4. If `graph_store`/`vector_store` is degraded, switch to safe mode by temporarily setting `KNOWLEDGE_BOARD_BACKEND=memory` and restarting.
5. Confirm recovery by replaying one smoke event through dashboard/event ingestion test.

## Snapshot Restore
To restore from latest snapshot:

```bash
python -m src.app --replay snapshots/snapshot_<step>.json --replay-start <step>
```

To inspect available snapshots and latest candidate:

```bash
ls snapshots/snapshot_*.json snapshots/snapshot_*.json.zst 2>/dev/null | tail -n 5
```

After restore, verify `/health` and compare expected current step in dashboard.

## Schema Migration
Snapshots are schema-versioned and validated on load. For migrations:
1. Back up snapshots:
   ```bash
   cp -r snapshots snapshots.backup.$(date +%Y%m%d%H%M%S)
   ```
2. Apply migration tooling/process for the new release.
3. Run targeted tests:
   ```bash
   python -m pytest tests/integration/test_snapshot_replay.py
   ```
4. Load a migrated snapshot in staging and verify `/ready` plus event ingestion.
5. Promote only after successful replay and health checks.
