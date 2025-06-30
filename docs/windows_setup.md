# Windows / WSL2 Setup

This guide walks through running the Culture.ai simulation on Windows using WSL2.
It covers enabling WSL2, installing Python 3.10, setting up Ollama, and running
the vertical slice example.

> **GPU Requirements**
> To utilize GPU acceleration you must run the simulation inside WSL2 with the
> [NVIDIA drivers for WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
> installed. After enabling WSL2 run `wsl --update` from an elevated PowerShell
> prompt to install the latest kernel and GPU support, then reboot when prompted.
> Without a supported GPU you can install the CPU-only PyTorch build.

## Enable WSL2

1. Open **PowerShell** as Administrator and run:
   ```powershell
   wsl --install
   ```
   This installs the WSL features and Ubuntu by default. Reboot if prompted.
2. Ensure WSL 2 is the default version:
   ```powershell
   wsl --set-default-version 2
   ```
3. Update WSL and install GPU support:
   ```powershell
   wsl --update
   ```
   Restart Windows when prompted to enable the latest kernel and GPU features.
   After reboot, run `wsl --shutdown` to apply the update and restart your
   distribution. You can verify GPU access inside WSL with `nvidia-smi`.

## Install Python 3.10

Inside your WSL distribution install Python 3.10 and the virtual environment
modules:

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-distutils
```

Create a virtual environment for the project:

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/macOS
# Windows:
.\venv\Scripts\activate.bat  # or .venv\Scripts\activate.bat
```
Activate the environment whenever you open a new terminal before running any
Python commands:

```cmd
.\venv\Scripts\activate.bat
```

Alternatively, run the helper script to automatically create `.venv` and
install all test requirements:

```cmd
scripts\setup_test_env.bat
```
This script uses `py -3.10 -m venv` to ensure it creates the environment with Python 3.10.

Copy the example environment file and customize it if needed:

```bash
cp .env.example .env
# Use `copy .env.example .env` from PowerShell
```

Edit `OLLAMA_API_BASE` in `.env` if your Ollama server uses a different URL.

If you plan to run Discord bots, also set `DISCORD_TOKENS_DB_URL` to your
PostgreSQL connection string and create the required table:

```sql
CREATE TABLE IF NOT EXISTS discord_tokens (
    agent_id TEXT PRIMARY KEY,
    token TEXT NOT NULL
);
```

## Install Ollama (≥0.1.34)

Culture.ai relies on the WSL build of **Ollama**. Install or upgrade it with:

```bash
curl https://ollama.ai/install.sh | sh
```

Verify you have version `0.1.34` or newer:

```bash
ollama --version
```

Expose port `11434` to your Windows host when launching Ollama so the Python
services can reach it. Pull the required model:

```bash
ollama pull mistral:latest
ollama serve &
```

Alternatively you can run the model with **vLLM**. Start the server from WSL or any
shell where Python is available:

```cmd
scripts\start_vllm.bat
```

This script mirrors `scripts/start_vllm.sh` and honors the `VLLM_MODEL`,
`VLLM_PORT`, and `VLLM_SWAP_SPACE` environment variables.

## Run the Example Vertical Slice

1. Install the CUDA-enabled PyTorch build and the remaining project requirements:
   ```bash
   pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.0+cu121
   pip install -r requirements.txt -r requirements-dev.txt
   ```
   The requirements file pins this PyTorch version, so ensure your system has CUDA 12.1 or a compatible driver installed.
   Install the NVIDIA drivers for WSL2 so the GPU is accessible from your Linux environment.
   If your Windows setup lacks a supported CUDA driver, install the CPU-only build instead:
   ```bash
   pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```
   You may also consult the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for alternate versions.
2. Execute the demo using your running Ollama instance:
   ```bash
   python -m examples.walking_vertical_slice
   ```
   On Windows you can instead run the convenience script:
   ```cmd
   scripts\vertical_slice.bat
   ```

   Before running the batch script copy `.env.example` to `.env` and set
   `OLLAMA_API_BASE` to your Ollama instance URL. Once **Issue&nbsp;1** is fixed
   the script will automatically load variables from `.env`.

The script checks for both `venv` and `.venv` directories and activates the first one it finds.

The script launches three agents for a few steps and stores their memories in
ChromaDB. See `docs/walking_vertical_slice.md` for details.

To verify the setup you can also run the integration test:

```bash
pytest tests/integration/agents/test_vertical_slice_real_llm.py -m integration
```

This test is skipped automatically if Ollama is not running.

## Run the Simulation

Once Ollama is running and your `.env` is configured, launch the main application:

```bash
python -m src.app --steps 5 --discord
```

The `--steps` flag sets how many iterations to perform before exiting. Use `--discord` to enable interaction with your configured Discord bot.

To enable deterministic replay of simulations, follow [docs/redpanda_setup.md](redpanda_setup.md) to install Redpanda and set the `ENABLE_REDPANDA` and `REDPANDA_BROKER` environment variables.

## Known Quirks and Limitations

- **`uvloop`**: The project uses `uvloop` for enhanced asyncio performance on POSIX-compliant systems (Linux, macOS). On Windows, it gracefully falls back to the standard `asyncio` event loop. Performance may differ slightly, but all functionality remains the same.
- **PyTorch**: If you are using features that require PyTorch (a dependency of `dspy-ai`), you may need to install the CPU-specific version if you do not have a compatible NVIDIA GPU. You can do this by adding `--index-url https://download.pytorch.org/whl/cpu` to your `pip install` commands. The project's extra-index-url for CUDA is defined in `requirements.in` but may not be suitable for all Windows environments.

This document is a work-in-progress. If you encounter other Windows-specific issues, please document them here.

