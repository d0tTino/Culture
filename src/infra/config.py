# src/infra/config.py
"""
Configuration module for Culture simulation.
Manages environment variables and configuration settings.
"""

import os
import logging
import dotenv

# Load environment variables from .env file if present
try:
    dotenv.load_dotenv()
    logging.debug("Loaded environment variables from .env file")
except Exception as e:
    logging.warning(f"Failed to load environment variables from .env file: {e}")

# --- Default Configuration Values ---
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # OpenAI API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Anthropic API key

# --- LLM Settings ---
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# --- Role Change Settings ---
ROLE_CHANGE_IP_COST = 5  # Cost in IP to request/confirm a role change
ROLE_CHANGE_COOLDOWN = 3 # Minimum number of steps an agent must stay in a role before requesting another change

# --- Data Units (DU) Settings ---
INITIAL_DATA_UNITS = 20  # Starting DU for new agents
ROLE_DU_GENERATION = {
    "Innovator": 2,     # Generates more DU per turn
    "Analyzer": 1,
    "Facilitator": 1,
    "Default Contributor": 0  # Default if role not found
}
PROPOSE_DETAILED_IDEA_DU_COST = 5  # DU cost for posting a detailed idea

# --- Discord Bot Settings ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID", "")
# Convert channel ID to int if it exists and is numeric
if DISCORD_CHANNEL_ID and DISCORD_CHANNEL_ID.isdigit():
    DISCORD_CHANNEL_ID = int(DISCORD_CHANNEL_ID)
else:
    DISCORD_CHANNEL_ID = None

# --- Helper Functions ---
def get(setting_name, default=None):
    """
    Retrieves a configuration setting by name.
    
    Args:
        setting_name (str): The name of the configuration setting to retrieve.
        default: The default value to return if the setting is not found.
        
    Returns:
        The configuration setting value, or the default if not found.
    """
    # Try to get from globals first
    setting = globals().get(setting_name)
    
    # If not in globals, try environment variables
    if setting is None:
        setting = os.getenv(setting_name, default)
    
    return setting

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
    dotenv.load_dotenv(dotenv_path=dotenv_path)
elif os.path.exists(dotenv_example_path):
    logger.warning(f".env file not found. Loading default settings from: {dotenv_example_path}")
    dotenv.load_dotenv(dotenv_path=dotenv_example_path)
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
# Log Discord settings, but hide the token
logger.info(f"  DISCORD_BOT_TOKEN: {'Set' if DISCORD_BOT_TOKEN else 'Not Set'}")
logger.info(f"  DISCORD_CHANNEL_ID: {DISCORD_CHANNEL_ID}")


# You can add functions here to validate configuration if needed
def get_redis_config():
    """Returns Redis connection details as a dictionary."""
    return {
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "db": REDIS_DB,
        "password": REDIS_PASSWORD
    } 