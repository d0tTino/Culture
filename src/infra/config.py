# src/infra/config.py
"""
Configuration loader for the Culture project.

Loads environment variables from a .env file located in the project's
config directory. Provides easy access to configuration settings.
"""

import os
from dotenv import load_dotenv
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine the path to the .env file (assuming it's in config/ relative to project root)
# This assumes the script is run from the project root or src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(project_root, 'config', '.env')
dotenv_example_path = os.path.join(project_root, 'config', '.env.example')

# Check if .env exists, otherwise try to load from .env.example as a fallback
if os.path.exists(dotenv_path):
    logger.info(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
elif os.path.exists(dotenv_example_path):
    logger.warning(f".env file not found. Loading default settings from: {dotenv_example_path}")
    load_dotenv(dotenv_path=dotenv_example_path)
else:
    logger.error(".env file not found and .env.example is also missing. Configuration may be incomplete.")
    # Optionally raise an error or exit if config is critical
    # raise FileNotFoundError("Configuration file (.env or .env.example) not found in config/ directory.")

# --- Accessor Functions or Variables ---

# Ollama Settings
OLLAMA_API_BASE = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434') # Default if not set

# Redis Settings
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379)) # Ensure port is integer
REDIS_DB = int(os.getenv('REDIS_DB', 0))       # Ensure db is integer
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None) # Default to None if not set

# --- Add other configuration variables as needed ---
# EXAMPLE_API_KEY = os.getenv('EXAMPLE_API_KEY')

# --- Log loaded configuration for verification (optional, be careful with sensitive data) ---
logger.info("Configuration loaded:")
logger.info(f"  OLLAMA_API_BASE: {OLLAMA_API_BASE}")
logger.info(f"  REDIS_HOST: {REDIS_HOST}")
logger.info(f"  REDIS_PORT: {REDIS_PORT}")
logger.info(f"  REDIS_DB: {REDIS_DB}")
# Avoid logging passwords directly:
logger.info(f"  REDIS_PASSWORD: {'Set' if REDIS_PASSWORD else 'Not Set'}")


# You can add functions here to validate configuration if needed
def get_redis_config():
    """Returns Redis connection details as a dictionary."""
    return {
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "db": REDIS_DB,
        "password": REDIS_PASSWORD
    } 