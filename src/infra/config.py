# src/infra/config.py
"""
Configuration module for Culture simulation.
Manages environment variables and configuration settings.
"""

import importlib
import logging
import os
from typing import Any, Optional

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file if present
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env file loading")
except Exception as e:
    logging.warning(f"Failed to load environment variables from .env file: {e}")

# Initialize module logger for configuration messages
logger = logging.getLogger(__name__)

# Allow runtime overrides of configuration values. Used in testing.
CONFIG_OVERRIDES: dict[str, Any] = {}

# Default values
DEFAULT_CONFIG: dict[str, object] = {
    "OLLAMA_API_BASE": "http://localhost:11434",
    "DEFAULT_LLM_MODEL": "mistral:latest",
    "DEFAULT_TEMPERATURE": 0.7,
    "MEMORY_THRESHOLD_L1": 0.2,
    "MEMORY_THRESHOLD_L2": 0.3,
    "VECTOR_STORE_DIR": "./chroma_db",
    "VECTOR_STORE_BACKEND": "chroma",  # chroma or weaviate
    "WEAVIATE_URL": "http://localhost:8080",
    "OLLAMA_REQUEST_TIMEOUT": 10,
    "TARGETED_MESSAGE_MULTIPLIER": 3.0,
    "POSITIVE_RELATIONSHIP_LEARNING_RATE": 0.3,
    "NEGATIVE_RELATIONSHIP_LEARNING_RATE": 0.4,
    "RELATIONSHIP_DECAY_FACTOR": 0.01,
    "MOOD_DECAY_FACTOR": 0.02,
    "ROLE_CHANGE_IP_COST": 5.0,
    "NEUTRAL_RELATIONSHIP_LEARNING_RATE": 0.1,
    "INITIAL_INFLUENCE_POINTS": 10.0,
    "INITIAL_DATA_UNITS": 20.0,
    "MAX_SHORT_TERM_MEMORY": 10,
    "SHORT_TERM_MEMORY_DECAY_RATE": 0.1,
    "MIN_RELATIONSHIP_SCORE": -1.0,
    "MAX_RELATIONSHIP_SCORE": 1.0,
    "MOOD_UPDATE_RATE": 0.2,
    "IP_COST_SEND_DIRECT_MESSAGE": 1.0,
    "DU_COST_PER_ACTION": 1.0,
    "ROLE_CHANGE_COOLDOWN": 3,
    "IP_AWARD_FOR_PROPOSAL": 5,
    "IP_COST_TO_POST_IDEA": 2,
    "DU_AWARD_FOR_PROPOSAL": 1,
    "IP_COST_CREATE_PROJECT": 10,
    "IP_COST_JOIN_PROJECT": 1,
    "IP_AWARD_FACILITATION_ATTEMPT": 3,
    "PROPOSE_DETAILED_IDEA_DU_COST": 5,
    "DU_AWARD_IDEA_ACKNOWLEDGED": 3,
    "DU_AWARD_SUCCESSFUL_ANALYSIS": 4,
    "DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE": 1,
    "DU_COST_DEEP_ANALYSIS": 3,
    "DU_COST_REQUEST_DETAILED_CLARIFICATION": 2,
    "DU_COST_CREATE_PROJECT": 10,
    "DU_COST_JOIN_PROJECT": 1,
    "MEMORY_PRUNING_L1_DELAY_STEPS": 10,
    "MEMORY_PRUNING_L2_MAX_AGE_DAYS": 30,
    "MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS": 100,
    "MEMORY_PRUNING_L1_MUS_THRESHOLD": 0.3,
    "MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION": 7,
    "MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS": 50,
    "MEMORY_PRUNING_L2_MUS_THRESHOLD": 0.25,
    "MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION": 14,
    "MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS": 150,
    "REDIS_HOST": "localhost",
    "REDIS_PORT": 6379,
    "REDIS_DB": 0,
    "DISCORD_BOT_TOKEN": "",
    "DISCORD_CHANNEL_ID": None,
    "DISCORD_TOKENS_DB_URL": "",
    "OPENAI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "DEFAULT_LOG_LEVEL": "INFO",
    "MEMORY_PRUNING_ENABLED": False,
    "MEMORY_PRUNING_L2_ENABLED": True,
    "MEMORY_PRUNING_L1_MUS_ENABLED": False,
    "MEMORY_PRUNING_L2_MUS_ENABLED": False,
    "MAX_PROJECT_MEMBERS": 3,
    "DEFAULT_MAX_SIMULATION_STEPS": 50,
    "MAX_KB_ENTRIES_FOR_PERCEPTION": 10,
    "MEMORY_STORE_TTL_SECONDS": 60 * 60 * 24 * 7,
    "MEMORY_STORE_PRUNE_INTERVAL_STEPS": 1,
    "MAX_IP_PER_TICK": 10.0,
    "MAX_DU_PER_TICK": 10.0,
}

# Global config dictionary
_CONFIG: dict[str, object] = {}

# Define keys that should be floats and ints for type conversion
FLOAT_CONFIG_KEYS = [
    "MEMORY_THRESHOLD_L1",
    "MEMORY_THRESHOLD_L2",
    "TARGETED_MESSAGE_MULTIPLIER",
    "POSITIVE_RELATIONSHIP_LEARNING_RATE",
    "NEGATIVE_RELATIONSHIP_LEARNING_RATE",
    "RELATIONSHIP_DECAY_FACTOR",
    "MOOD_DECAY_FACTOR",
    "DEFAULT_TEMPERATURE",
    "MEMORY_PRUNING_L1_MUS_THRESHOLD",
    "MEMORY_PRUNING_L2_MUS_THRESHOLD",
    "NEUTRAL_RELATIONSHIP_LEARNING_RATE",
    "INITIAL_INFLUENCE_POINTS",
    "INITIAL_DATA_UNITS",
    "SHORT_TERM_MEMORY_DECAY_RATE",
    "MIN_RELATIONSHIP_SCORE",
    "MAX_RELATIONSHIP_SCORE",
    "MOOD_UPDATE_RATE",
    "IP_COST_SEND_DIRECT_MESSAGE",
    "DU_COST_PER_ACTION",
    "ROLE_CHANGE_IP_COST",
    "MAX_IP_PER_TICK",
    "MAX_DU_PER_TICK",
]
INT_CONFIG_KEYS = [
    "IP_AWARD_FOR_PROPOSAL",
    "IP_COST_TO_POST_IDEA",
    "DU_AWARD_FOR_PROPOSAL",
    "IP_COST_CREATE_PROJECT",
    "IP_COST_JOIN_PROJECT",
    "IP_AWARD_FACILITATION_ATTEMPT",
    "PROPOSE_DETAILED_IDEA_DU_COST",
    "DU_AWARD_IDEA_ACKNOWLEDGED",
    "DU_AWARD_SUCCESSFUL_ANALYSIS",
    "DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE",
    "DU_COST_DEEP_ANALYSIS",
    "DU_COST_REQUEST_DETAILED_CLARIFICATION",
    "DU_COST_CREATE_PROJECT",
    "DU_COST_JOIN_PROJECT",
    "MEMORY_PRUNING_L1_DELAY_STEPS",
    "MEMORY_PRUNING_L2_MAX_AGE_DAYS",
    "MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS",
    "MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION",
    "MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS",
    "MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION",
    "MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS",
    "REDIS_PORT",
    "REDIS_DB",
    "MAX_SHORT_TERM_MEMORY",
    "ROLE_CHANGE_COOLDOWN",
    "OLLAMA_REQUEST_TIMEOUT",
    "MAX_PROJECT_MEMBERS",
    "DEFAULT_MAX_SIMULATION_STEPS",
    "MAX_KB_ENTRIES_FOR_PERCEPTION",
    "MEMORY_STORE_TTL_SECONDS",
    "MEMORY_STORE_PRUNE_INTERVAL_STEPS",
]
BOOL_CONFIG_KEYS = [
    "MEMORY_PRUNING_ENABLED",
    "MEMORY_PRUNING_L2_ENABLED",
    "MEMORY_PRUNING_L1_MUS_ENABLED",
    "MEMORY_PRUNING_L2_MUS_ENABLED",
]


def load_config() -> dict[str, object]:
    """
    Load configuration from environment variables.

    Returns:
        dict[str, object]: The loaded configuration dictionary.
    """
    global _CONFIG

    # Start with default config
    _CONFIG = DEFAULT_CONFIG.copy()

    # Process all keys that might be set in environment, converting known types
    # Iterate over a combined set of keys: DEFAULT_CONFIG keys + known float/int keys
    # This ensures that if an env var is set for a known float/int key not in DEFAULT_CONFIG,
    # it's still processed and type-converted.
    all_potential_keys = (
        set(DEFAULT_CONFIG.keys())
        | set(FLOAT_CONFIG_KEYS)
        | set(INT_CONFIG_KEYS)
        | set(BOOL_CONFIG_KEYS)
    )

    for key in all_potential_keys:
        env_value = os.environ.get(key)
        if env_value:
            try:
                if key in FLOAT_CONFIG_KEYS:
                    _CONFIG[key] = float(env_value)
                elif key in INT_CONFIG_KEYS:
                    # Special handling for DISCORD_CHANNEL_ID as it can be None
                    if key == "DISCORD_CHANNEL_ID":
                        if env_value.isdigit():
                            _CONFIG[key] = int(env_value)
                        else:
                            _CONFIG[key] = None  # Or handle as error/warning
                            logger.warning(
                                f"DISCORD_CHANNEL_ID '{env_value}' is not a digit. Setting to None."
                            )
                    else:
                        _CONFIG[key] = int(env_value)
                elif key in BOOL_CONFIG_KEYS:
                    _CONFIG[key] = env_value.lower() in ("true", "1", "t", "yes", "y")
                else:  # For keys in DEFAULT_CONFIG but not explicitly float/int (e.g. strings)
                    _CONFIG[key] = env_value
            except ValueError:
                logger.warning(
                    f"Could not convert env var {key}='{env_value}' to its target type. "
                    f"Using default: {_CONFIG.get(key, DEFAULT_CONFIG.get(key))}"
                )
        # If env_value is not set, but key is in DEFAULT_CONFIG, it's already set from the copy.
        # If env_value is not set and key is a known float/int key NOT in DEFAULT_CONFIG,
        # we might want to explicitly set it to None or a sensible default if not already covered.
        # However, AgentState fields using these should have their own defaults or handle None.
        # For now, if not in env and not in DEFAULT_CONFIG, it won't be in _CONFIG.
        # Pydantic fields in AgentState will use their `default_factory=lambda: get_config(KEY)`
        # which will then pull from DEFAULT_CONFIG if get_config(KEY) from _CONFIG is None.

    # Special handling for ROLE_DU_GENERATION if it can be overridden by ENV
    # For now, it's taken from DEFAULT_CONFIG. If it needs to be env-configurable,
    # it would require parsing a JSON string from the env var.
    _CONFIG["ROLE_DU_GENERATION"] = {  # Define it directly here based on DEFAULT_CONFIG structure
        "Facilitator": {
            "base": _CONFIG.get("ROLE_DU_GENERATION_FACILITATOR_BASE", 1.0),
            "bonus_factor": _CONFIG.get("ROLE_DU_GENERATION_FACILITATOR_BONUS", 0.0),
        },
        "Innovator": {
            "base": _CONFIG.get("ROLE_DU_GENERATION_INNOVATOR_BASE", 1.0),
            "bonus_factor": _CONFIG.get("ROLE_DU_GENERATION_INNOVATOR_BONUS", 0.5),
        },
        "Analyzer": {
            "base": _CONFIG.get("ROLE_DU_GENERATION_ANALYZER_BASE", 1.0),
            "bonus_factor": _CONFIG.get("ROLE_DU_GENERATION_ANALYZER_BONUS", 0.5),
        },
    }

    # Log minimal information to avoid leaking sensitive values
    logger.info("Configuration loaded")
    # Fail fast if critical config is missing
    if not _CONFIG.get("OLLAMA_API_BASE"):
        logger.critical("OLLAMA_API_BASE is missing from configuration. Exiting.")
        import sys

        sys.exit(1)
    return _CONFIG


def get_config(key: Optional[str] = None) -> Any:
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
load_config()

OLLAMA_API_BASE = get_config("OLLAMA_API_BASE")
OLLAMA_REQUEST_TIMEOUT = get_config("OLLAMA_REQUEST_TIMEOUT")
DEFAULT_LLM_MODEL = get_config("DEFAULT_LLM_MODEL")
MEMORY_THRESHOLD_L1 = get_config("MEMORY_THRESHOLD_L1")
MEMORY_THRESHOLD_L2 = get_config("MEMORY_THRESHOLD_L2")
VECTOR_STORE_DIR = get_config("VECTOR_STORE_DIR")
VECTOR_STORE_BACKEND = get_config("VECTOR_STORE_BACKEND")
WEAVIATE_URL = get_config("WEAVIATE_URL")

# --- Basic Configuration ---
DEFAULT_LOG_LEVEL = get_config("DEFAULT_LOG_LEVEL")

# --- API Keys ---
OPENAI_API_KEY = get_config("OPENAI_API_KEY")
ANTHROPIC_API_KEY = get_config("ANTHROPIC_API_KEY")

# --- LLM Settings ---
DEFAULT_TEMPERATURE = get_config("DEFAULT_TEMPERATURE")

# --- Ollama Settings ---

# --- Redis Settings ---
REDIS_HOST = get_config("REDIS_HOST")
REDIS_PORT = get_config("REDIS_PORT")
REDIS_DB = get_config("REDIS_DB")
REDIS_PASSWORD = get_config("REDIS_PASSWORD")

# --- Discord Bot Settings ---
DISCORD_BOT_TOKEN = get_config("DISCORD_BOT_TOKEN")
DISCORD_CHANNEL_ID = get_config("DISCORD_CHANNEL_ID")
DISCORD_TOKENS_DB_URL = get_config("DISCORD_TOKENS_DB_URL")

# --- Memory Pruning Settings ---
# Whether to enable automatic memory pruning (default to False for safety)
MEMORY_PRUNING_ENABLED = bool(get_config("MEMORY_PRUNING_ENABLED"))
# How many steps to wait after a Level 2 summary before pruning the Level 1 summaries
MEMORY_PRUNING_L1_DELAY_STEPS = get_config("MEMORY_PRUNING_L1_DELAY_STEPS")
# Whether to enable Level 2 summary pruning
MEMORY_PRUNING_L2_ENABLED = bool(get_config("MEMORY_PRUNING_L2_ENABLED"))
# Maximum age in days for an L2 summary before it's considered for pruning
MEMORY_PRUNING_L2_MAX_AGE_DAYS = get_config("MEMORY_PRUNING_L2_MAX_AGE_DAYS")
# How many simulation steps occur between L2 pruning checks
MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS = get_config("MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS")

# --- MUS-based L1 Pruning Settings ---
MEMORY_PRUNING_L1_MUS_ENABLED = bool(get_config("MEMORY_PRUNING_L1_MUS_ENABLED"))
MEMORY_PRUNING_L1_MUS_THRESHOLD = get_config("MEMORY_PRUNING_L1_MUS_THRESHOLD")
MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION = get_config(
    "MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION"
)
MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS = get_config(
    "MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS"
)

# --- MUS-based L2 Pruning Settings ---
MEMORY_PRUNING_L2_MUS_ENABLED = bool(get_config("MEMORY_PRUNING_L2_MUS_ENABLED"))
MEMORY_PRUNING_L2_MUS_THRESHOLD = get_config("MEMORY_PRUNING_L2_MUS_THRESHOLD")
MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION = get_config(
    "MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION"
)
MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS = get_config(
    "MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS"
)

# --- Mood and Relationship Settings ---
MOOD_DECAY_FACTOR = get_config("MOOD_DECAY_FACTOR")
RELATIONSHIP_DECAY_FACTOR = get_config("RELATIONSHIP_DECAY_FACTOR")
POSITIVE_RELATIONSHIP_LEARNING_RATE = get_config("POSITIVE_RELATIONSHIP_LEARNING_RATE")
NEGATIVE_RELATIONSHIP_LEARNING_RATE = get_config("NEGATIVE_RELATIONSHIP_LEARNING_RATE")
NEUTRAL_RELATIONSHIP_LEARNING_RATE = get_config("NEUTRAL_RELATIONSHIP_LEARNING_RATE")
TARGETED_MESSAGE_MULTIPLIER = get_config("TARGETED_MESSAGE_MULTIPLIER")

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
    get_config("INITIAL_INFLUENCE_POINTS")
)  # Starting IP for new agents
IP_AWARD_FOR_PROPOSAL = int(
    get_config("IP_AWARD_FOR_PROPOSAL")
)  # Amount of IP awarded for successfully proposing an idea
IP_COST_TO_POST_IDEA = get_config("IP_COST_TO_POST_IDEA")  # Cost in IP to post an idea
ROLE_CHANGE_IP_COST = int(
    get_config("ROLE_CHANGE_IP_COST")
)  # Cost in IP to request/confirm a role change
IP_COST_CREATE_PROJECT = int(
    get_config("IP_COST_CREATE_PROJECT")
)  # Cost in IP to create a new project
IP_COST_JOIN_PROJECT = int(
    get_config("IP_COST_JOIN_PROJECT")
)  # Cost in IP to join an existing project
IP_COST_SEND_DIRECT_MESSAGE = int(
    get_config("IP_COST_SEND_DIRECT_MESSAGE")
)  # Cost in IP to send a direct message
IP_AWARD_FACILITATION_ATTEMPT = int(
    get_config("IP_AWARD_FACILITATION_ATTEMPT")
)  # IP award for attempting facilitation

# --- Data Units (DU) Settings ---
INITIAL_DATA_UNITS = get_config("INITIAL_DATA_UNITS")  # Starting DU for new agents
PROPOSE_DETAILED_IDEA_DU_COST = int(
    get_config("PROPOSE_DETAILED_IDEA_DU_COST")
)  # DU cost for posting a detailed idea
DU_AWARD_IDEA_ACKNOWLEDGED = int(
    get_config("DU_AWARD_IDEA_ACKNOWLEDGED")
)  # DU awarded to original proposer if idea is referenced
DU_AWARD_SUCCESSFUL_ANALYSIS = int(
    get_config("DU_AWARD_SUCCESSFUL_ANALYSIS")
)  # DU awarded to Analyzer for useful critique
DU_AWARD_FOR_PROPOSAL = int(
    get_config("DU_AWARD_FOR_PROPOSAL")
)  # DU awarded when an agent proposes an idea
DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE = int(
    get_config("DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE")
)  # DU bonus for referencing a board entry
DU_COST_DEEP_ANALYSIS = int(
    get_config("DU_COST_DEEP_ANALYSIS")
)  # Cost for an Analyzer to perform a "deep analysis"
DU_COST_REQUEST_DETAILED_CLARIFICATION = int(
    get_config("DU_COST_REQUEST_DETAILED_CLARIFICATION")
)  # DU cost for asking for detailed clarification
DU_COST_CREATE_PROJECT = int(
    get_config("DU_COST_CREATE_PROJECT")
)  # Cost in DU to create a new project
DU_COST_JOIN_PROJECT = int(
    get_config("DU_COST_JOIN_PROJECT")
)  # Cost in DU to join an existing project

# --- Role Settings ---
ROLE_DU_GENERATION = {
    "Facilitator": {"base": 1.0, "bonus_factor": 0.0},
    "Innovator": {"base": 1.0, "bonus_factor": 0.5},
    "Analyzer": {"base": 1.0, "bonus_factor": 0.5},
    # Add other roles here if they have different DU generation patterns
}
ROLE_CHANGE_COOLDOWN = int(
    get_config("ROLE_CHANGE_COOLDOWN")
)  # Min steps an agent must stay in a role

# --- Project Settings ---
MAX_PROJECT_MEMBERS = int(
    get_config("MAX_PROJECT_MEMBERS")
)  # Maximum number of members in a project

# Vector Store Configuration
WEAVIATE_URL = get_config("WEAVIATE_URL")
CHROMA_DB_PATH = get_config("CHROMA_DB_PATH")
DEFAULT_VECTOR_STORE = get_config("DEFAULT_VECTOR_STORE")  # "chroma" or "weaviate"

# Knowledge Board
MAX_KB_ENTRIES_FOR_PERCEPTION = get_config("MAX_KB_ENTRIES_FOR_PERCEPTION")

# Simulation Parameters
DEFAULT_MAX_SIMULATION_STEPS = get_config("DEFAULT_MAX_SIMULATION_STEPS")
MEMORY_STORE_TTL_SECONDS = get_config("MEMORY_STORE_TTL_SECONDS")
MEMORY_STORE_PRUNE_INTERVAL_STEPS = get_config("MEMORY_STORE_PRUNE_INTERVAL_STEPS")
MAX_IP_PER_TICK = get_config("MAX_IP_PER_TICK")
MAX_DU_PER_TICK = get_config("MAX_DU_PER_TICK")


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


# Log loaded configuration for verification. Avoid including sensitive data.
logger.info("Configuration loaded successfully")


def get_config_value_with_override(
    key: str, default: Any = None, module_name: str = "src.infra.config_default"
) -> Any:
    """Fetches a config value, trying overrides first, then primary, then default."""
    # Dynamically import the configuration module
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.error(f"Failed to import configuration module: {module_name}")
        return default

    # Try to get from overrides
    if CONFIG_OVERRIDES.get(key) is not None:
        # logger.debug(f"Config key '{key}' found in overrides.")
        return CONFIG_OVERRIDES[key]
    else:
        # If not in overrides, try to get from the primary config module
        primary_value = getattr(module, str(key), None)  # Ensure key is string for getattr
        if primary_value is not None:
            # logger.debug(f"Config key '{key}' found in primary config module '{module_name}'.")
            return primary_value
        else:
            # If not in primary, return default
            # logger.debug(f"Config key '{key}' not found. Returning default value: {default}")
            return default
