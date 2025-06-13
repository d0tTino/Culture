# Windows / WSL2 Setup

This guide walks through running the Culture.ai simulation on Windows using WSL2.
It covers enabling WSL2, installing Python 3.10, setting up Ollama, and running
the vertical slice example.

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
source venv/bin/activate
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

## Run the Example Vertical Slice

1. Install project requirements from the repository root:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
2. Execute the demo using your running Ollama instance:
   ```bash
   python -m examples.walking_vertical_slice
   ```

The script launches three agents for a few steps and stores their memories in
ChromaDB. See `docs/walking_vertical_slice.md` for details.
