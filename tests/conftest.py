"""
Pytest fixtures for use across test files.
"""

# Import warning filters to suppress third-party dependency warnings
# This needs to be done early before other imports
import os
import pathlib
import shutil
import sys
import tempfile

# Add the project root to path to allow importing src modules
project_root = str(pathlib.Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our warning filters
from src.infra.warning_filters import configure_warning_filters

configure_warning_filters()  # Apply filters

from unittest.mock import patch

import pytest

from tests.utils.mock_llm import MockLLM


def is_ollama_running():
    """Check if Ollama server is running by attempting to connect to localhost:11434"""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)  # Short timeout for quick check
        s.connect(("localhost", 11434))
        s.close()
        return True
    except (OSError, socket.timeout):
        return False


# Mark for tests requiring Ollama
require_ollama = pytest.mark.skipif(
    not is_ollama_running(), reason="Ollama server is not running on localhost:11434"
)


@pytest.fixture
def mock_llm_client():
    """Fixture to provide a mocked LLMClient"""
    with patch("src.infra.llm_client.LLMClient") as MockLLMClient:
        mock_client = MockLLM()
        MockLLMClient.return_value = mock_client
        yield MockLLMClient


@pytest.fixture(autouse=True)
def mock_ollama_by_default(request, monkeypatch):
    """
    Automatically mock Ollama functions unless the test is explicitly marked with require_ollama.

    This prevents 404 errors when Ollama is not running.
    """
    # Skip mocking if test is marked with require_ollama
    if request.node.get_closest_marker("require_ollama"):
        # If marked with require_ollama but server isn't running, the test will be skipped
        # by the require_ollama mark, so we don't need to do anything here
        return

    # If test is not explicitly requiring Ollama, mock all Ollama functions
    from src.infra.llm_mock_helper import patch_ollama_functions

    patch_ollama_functions(monkeypatch)


@pytest.fixture(scope="session")
def chroma_test_dir(request):
    """
    Provides a unique ChromaDB test directory for each pytest-xdist worker.
    On Linux, uses /dev/shm/chroma_tests/{worker_id}/ for tmpfs speed.
    On other OSs, falls back to a temp directory.
    Auto-deletes the directory after the session.
    """
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    if sys.platform.startswith("linux") and os.path.exists("/dev/shm"):
        base_dir = f"/dev/shm/chroma_tests/{worker_id}"
    else:
        base_dir = tempfile.mkdtemp(prefix=f"chroma_tests_{worker_id}_")
    os.makedirs(base_dir, exist_ok=True)
    yield base_dir
    # Teardown: remove the directory after the session
    try:
        shutil.rmtree(base_dir)
    except Exception as e:
        print(f"Warning: Failed to remove Chroma test dir {base_dir}: {e}")
