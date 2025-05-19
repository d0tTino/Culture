# src/infra/config.py
"""
Configuration module for Culture simulation.
Manages environment variables and configuration settings.
"""

import json
import logging
import os
from typing import Optional

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file if present
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env file loading")
except Exception as e:
    logging.warning(f"Failed to load environment variables from .env file: {e}")

# Configure logger
logger = logging.getLogger(__name__)

# Default values
DEFAULT_CONFIG: dict[str, object] = {
    "OLLAMA_API_BASE": "http://localhost:11434",
    "DEFAULT_LLM_MODEL": "mistral:latest",
    "MEMORY_THRESHOLD_L1": 0.2,
    "MEMORY_THRESHOLD_L2": 0.3,
    "VECTOR_STORE_DIR": "./chroma_db",
    "VECTOR_STORE_BACKEND": "chroma",  # chroma or weaviate
    "WEAVIATE_URL": "http://localhost:8080",
}

# Global config dictionary
_CONFIG: dict[str, object] = {}


def load_config() -> dict[str, object]:
    """
    Load configuration from environment variables.

    Returns:
        dict[str, object]: The loaded configuration dictionary.
    """
    global _CONFIG

    # Start with default config
    _CONFIG = DEFAULT_CONFIG.copy()

    # Override with environment variables
    for key in DEFAULT_CONFIG:
        try:
            env_value = os.environ.get(key)
            if env_value:
                # Convert string values to appropriate types
                if key in ["MEMORY_THRESHOLD_L1", "MEMORY_THRESHOLD_L2"]:
                    try:
                        _CONFIG[key] = float(env_value)
                    except ValueError:
                        logger.warning(
                            f"Could not convert {key}={env_value} to float. "
                            f"Using default {_CONFIG[key]}"
                        )
                else:
                    _CONFIG[key] = env_value
        except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config key {key}: {e}", exc_info=True)

    # Add support for VECTOR_STORE_BACKEND and WEAVIATE_URL
    for key in ["VECTOR_STORE_BACKEND", "WEAVIATE_URL"]:
        env_value = os.environ.get(key)
        if env_value:
            _CONFIG[key] = env_value

    logger.info(f"Configuration loaded: {_CONFIG}")
    # Fail fast if critical config is missing
    if not _CONFIG.get("OLLAMA_API_BASE"):
        logger.critical("OLLAMA_API_BASE is missing from configuration. Exiting.")
        import sys

        sys.exit(1)
    return _CONFIG


def get_config(key: Optional[str] = None) -> object:
    """
    Get a configuration value.

    Args:
        key (str, optional): The configuration key to retrieve. If None, returns the entire
            config dict.

    Returns:
        object: The configuration value or the entire config dict.
    """
    if not _CONFIG:
        load_config()

    if key is None:
        return _CONFIG

    return _CONFIG.get(key, DEFAULT_CONFIG.get(key))


# Initialize the configuration on module import
OLLAMA_API_BASE = get_config("OLLAMA_API_BASE")
DEFAULT_LLM_MODEL = get_config("DEFAULT_LLM_MODEL")
MEMORY_THRESHOLD_L1 = get_config("MEMORY_THRESHOLD_L1")
MEMORY_THRESHOLD_L2 = get_config("MEMORY_THRESHOLD_L2")
VECTOR_STORE_DIR = get_config("VECTOR_STORE_DIR")
VECTOR_STORE_BACKEND = get_config("VECTOR_STORE_BACKEND")
WEAVIATE_URL = get_config("WEAVIATE_URL")

# --- Basic Configuration ---
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # OpenAI API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Anthropic API key

# --- LLM Settings ---
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# --- Ollama Settings ---

# --- Redis Settings ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# --- Discord Bot Settings ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID: object = os.getenv("DISCORD_CHANNEL_ID", "")
# Convert channel ID to int if it exists and is numeric
if DISCORD_CHANNEL_ID and isinstance(DISCORD_CHANNEL_ID, str) and DISCORD_CHANNEL_ID.isdigit():
    DISCORD_CHANNEL_ID = int(DISCORD_CHANNEL_ID)
else:
    DISCORD_CHANNEL_ID = None

# --- Memory Pruning Settings ---
# Whether to enable automatic memory pruning (default to False for safety)
MEMORY_PRUNING_ENABLED = os.getenv("MEMORY_PRUNING_ENABLED", "False").lower() == "true"
# How many steps to wait after a Level 2 summary before pruning the Level 1 summaries
MEMORY_PRUNING_L1_DELAY_STEPS = int(os.getenv("MEMORY_PRUNING_L1_DELAY_STEPS", "10"))
# Whether to enable Level 2 summary pruning
MEMORY_PRUNING_L2_ENABLED = os.getenv("MEMORY_PRUNING_L2_ENABLED", "True").lower() == "true"
# Maximum age in days for an L2 summary before it's considered for pruning
MEMORY_PRUNING_L2_MAX_AGE_DAYS = int(os.getenv("MEMORY_PRUNING_L2_MAX_AGE_DAYS", "30"))
# How many simulation steps occur between L2 pruning checks
MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS = int(
    os.getenv("MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS", "100")
)

# --- MUS-based L1 Pruning Settings ---
MEMORY_PRUNING_L1_MUS_ENABLED = (
    os.getenv("MEMORY_PRUNING_L1_MUS_ENABLED", "False").lower() == "true"
)
MEMORY_PRUNING_L1_MUS_THRESHOLD = float(os.getenv("MEMORY_PRUNING_L1_MUS_THRESHOLD", "0.3"))
MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION = int(
    os.getenv("MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION", "7")
)
MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS = int(
    os.getenv("MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS", "50")
)

# --- MUS-based L2 Pruning Settings ---
MEMORY_PRUNING_L2_MUS_ENABLED = (
    os.getenv("MEMORY_PRUNING_L2_MUS_ENABLED", "False").lower() == "true"
)
MEMORY_PRUNING_L2_MUS_THRESHOLD = float(os.getenv("MEMORY_PRUNING_L2_MUS_THRESHOLD", "0.25"))
MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION = int(
    os.getenv("MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION", "14")
)
MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS = int(
    os.getenv("MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS", "150")
)

# --- Mood and Relationship Settings ---
MOOD_DECAY_FACTOR = float(
    os.getenv("MOOD_DECAY_FACTOR", "0.02")
)  # Mood decays towards neutral by 2% each turn
RELATIONSHIP_DECAY_FACTOR = float(
    os.getenv("RELATIONSHIP_DECAY_FACTOR", "0.01")
)  # Relationships decay towards neutral by 1% each turn
POSITIVE_RELATIONSHIP_LEARNING_RATE = float(
    os.getenv("POSITIVE_RELATIONSHIP_LEARNING_RATE", "0.3")
)  # Learning rate for positive sentiment interactions
NEGATIVE_RELATIONSHIP_LEARNING_RATE = float(
    os.getenv("NEGATIVE_RELATIONSHIP_LEARNING_RATE", "0.4")
)  # Learning rate for negative sentiment interactions
NEUTRAL_RELATIONSHIP_LEARNING_RATE = float(
    os.getenv("NEUTRAL_RELATIONSHIP_LEARNING_RATE", "0.1")
)  # Learning rate for neutral sentiment interactions
TARGETED_MESSAGE_MULTIPLIER = float(
    os.getenv("TARGETED_MESSAGE_MULTIPLIER", "3.0")
)  # Multiplier for relationship changes from targeted messages vs broadcasts

# --- Relationship Label Mapping ---
RELATIONSHIP_LABELS = {
    (-1.0, -0.7): "Hostile",
    (-0.7, -0.4): "Negative",
    (-0.4, -0.1): "Cautious",
    (-0.1, 0.1): "Neutral",
    (0.1, 0.4): "Cordial",
    (0.4, 0.7): "Positive",
    (0.7, 1.0): "Allied",
}

# --- Sentiment Numeric Mapping ---
SENTIMENT_TO_NUMERIC = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# --- Influence Points (IP) Settings ---
INITIAL_INFLUENCE_POINTS = int(
    os.getenv("INITIAL_INFLUENCE_POINTS", "10")
)  # Starting IP for new agents
IP_AWARD_FOR_PROPOSAL = int(
    os.getenv("IP_AWARD_FOR_PROPOSAL", "5")
)  # Amount of IP awarded for successfully proposing an idea
IP_COST_TO_POST_IDEA = int(os.getenv("IP_COST_TO_POST_IDEA", "2"))  # Cost in IP to post an idea
ROLE_CHANGE_IP_COST = int(
    os.getenv("ROLE_CHANGE_IP_COST", "5")
)  # Cost in IP to request/confirm a role change
IP_COST_CREATE_PROJECT = int(
    os.getenv("IP_COST_CREATE_PROJECT", "10")
)  # Cost in IP to create a new project
IP_COST_JOIN_PROJECT = int(
    os.getenv("IP_COST_JOIN_PROJECT", "1")
)  # Cost in IP to join an existing project

# --- Data Units (DU) Settings ---
INITIAL_DATA_UNITS = int(os.getenv("INITIAL_DATA_UNITS", "20"))  # Starting DU for new agents
PROPOSE_DETAILED_IDEA_DU_COST = int(
    os.getenv("PROPOSE_DETAILED_IDEA_DU_COST", "5")
)  # DU cost for posting a detailed idea
DU_AWARD_IDEA_ACKNOWLEDGED = int(
    os.getenv("DU_AWARD_IDEA_ACKNOWLEDGED", "3")
)  # DU awarded to original proposer if idea is referenced
DU_AWARD_SUCCESSFUL_ANALYSIS = int(
    os.getenv("DU_AWARD_SUCCESSFUL_ANALYSIS", "4")
)  # DU awarded to Analyzer for useful critique
DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE = int(
    os.getenv("DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE", "1")
)  # DU bonus for referencing a board entry
DU_COST_DEEP_ANALYSIS = int(
    os.getenv("DU_COST_DEEP_ANALYSIS", "3")
)  # Cost for an Analyzer to perform a "deep analysis"
DU_COST_REQUEST_DETAILED_CLARIFICATION = int(
    os.getenv("DU_COST_REQUEST_DETAILED_CLARIFICATION", "2")
)  # Cost to ask detailed clarification
DU_COST_CREATE_PROJECT = int(
    os.getenv("DU_COST_CREATE_PROJECT", "10")
)  # Cost in DU to create a new project
DU_COST_JOIN_PROJECT = int(
    os.getenv("DU_COST_JOIN_PROJECT", "1")
)  # Cost in DU to join an existing project

# --- Role Settings ---
ROLE_DU_GENERATION = {
    "Innovator": int(os.getenv("ROLE_DU_GENERATION_INNOVATOR", "2")),
    "Analyzer": int(os.getenv("ROLE_DU_GENERATION_ANALYZER", "1")),
    "Facilitator": int(os.getenv("ROLE_DU_GENERATION_FACILITATOR", "1")),
    "Default Contributor": int(os.getenv("ROLE_DU_GENERATION_DEFAULT", "0")),
}
ROLE_CHANGE_COOLDOWN = int(
    os.getenv("ROLE_CHANGE_COOLDOWN", "3")
)  # Min steps an agent must stay in a role

# --- Project Settings ---
MAX_PROJECT_MEMBERS = int(
    os.getenv("MAX_PROJECT_MEMBERS", "3")
)  # Maximum number of members in a project


# --- Helper Functions ---
def get(setting_name: str, default: Optional[str] = None) -> object:
    """
    Retrieves a configuration setting by name.

    Args:
        setting_name (str): The name of the configuration setting to retrieve.
        default: The default value to return if the setting is not found.

    Returns:
        object: The configuration setting value, or the default if not found.
    """
    # Try to get from globals first
    setting = globals().get(setting_name)

    # If not in globals, try environment variables
    if setting is None:
        setting = os.getenv(setting_name, default)

    return setting


def get_relationship_label(score: float) -> str:
    """
    Returns a descriptive relationship label based on a relationship score.

    Args:
        score (float): The relationship score (-1.0 to 1.0)

    Returns:
        str: A descriptive relationship label
    """
    for (min_val, max_val), label in RELATIONSHIP_LABELS.items():
        if min_val <= score <= max_val:
            return label
    return "Neutral"  # Default fallback


def get_redis_config() -> dict[str, object]:
    """Returns Redis connection details as a dictionary."""
    return {"host": REDIS_HOST, "port": REDIS_PORT, "db": REDIS_DB, "password": REDIS_PASSWORD}


# Configure basic logging
logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL), format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log loaded configuration for verification (optional, be careful with sensitive data)
logger.info("Configuration loaded successfully")
