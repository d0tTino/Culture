# Walking Vertical Slice

This example demonstrates a minimal end-to-end run of the Culture.ai simulation using a real LLM via Ollama. It is useful for verifying your local setup.

## Prerequisites
- Python environment with project dependencies installed
- [Ollama](https://ollama.ai/) running locally with the `mistral:latest` model pulled
- Environment variable `OLLAMA_API_BASE` set to your Ollama server URL (e.g. `http://localhost:11434`)
 - Copy `.env.example` to `.env` and ensure the above variable is defined there

## Running the Demo
1. Activate your virtual environment and install the requirements:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
2. Pull the model and ensure Ollama is running:
   ```bash
   ollama pull mistral:latest
   ollama serve &  # if not already running
   ```
3. Execute the demo script:
   ```bash
   python -m examples.walking_vertical_slice
   ```

The script spins up two agents and runs three steps of the simulation. All LLM calls go through your local Ollama instance; no mocking is applied.
