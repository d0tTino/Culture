"""Configuration utilities for the Culture project."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from .settings import ConfigSettings, settings

logger = logging.getLogger(__name__)

# Allow runtime overrides in tests
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
    # Knowledge board backend: "memory" or "graph"
    "KNOWLEDGE_BOARD_BACKEND": "memory",
    # Neo4j connection details for graph-backed knowledge board
    "GRAPH_DB_URI": "bolt://localhost:7687",
    "GRAPH_DB_USER": "neo4j",
    "GRAPH_DB_PASSWORD": "test",
    "WEAVIATE_URL": "http://localhost:8080",
    "OLLAMA_REQUEST_TIMEOUT": 10,
    "OPA_URL": "",
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
    "LAW_PASS_IP_REWARD": 1,
    "LAW_PASS_DU_REWARD": 0,
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
    # Map action economic settings
    "MAP_MOVE_IP_COST": 0.0,
    "MAP_MOVE_IP_REWARD": 0.0,
    "MAP_MOVE_DU_COST": 0.0,
    "MAP_MOVE_DU_REWARD": 0.0,
    "MAP_GATHER_IP_COST": 0.0,
    "MAP_GATHER_IP_REWARD": 0.0,
    "MAP_GATHER_DU_COST": 0.0,
    "MAP_GATHER_DU_REWARD": 0.0,
    "MAP_BUILD_IP_COST": 0.0,
    "MAP_BUILD_IP_REWARD": 0.0,
    "MAP_BUILD_DU_COST": 0.0,
    "MAP_BUILD_DU_REWARD": 0.0,
    "MEMORY_PRUNING_L1_DELAY_STEPS": 10,
    "MEMORY_PRUNING_L2_MAX_AGE_DAYS": 30,
    "MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS": 100,
    "MEMORY_PRUNING_L1_MUS_THRESHOLD": 0.3,
    "MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION": 7,
    "MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS": 50,
    "MEMORY_PRUNING_L2_MUS_THRESHOLD": 0.25,
    "MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION": 14,
    "MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS": 150,
    "MEMORY_PRUNING_USAGE_COUNT_THRESHOLD": 3,
    "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS": 0,
    "REDIS_HOST": "localhost",
    "REDIS_PORT": 6379,
    "REDIS_DB": 0,
    "DISCORD_BOT_TOKEN": "",
    "DISCORD_CHANNEL_ID": None,
    "DISCORD_TOKENS_DB_URL": "",
    "OPENAI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "DEFAULT_LOG_LEVEL": "INFO",
    # Endpoint for sending logs via OpenTelemetry
    "OTEL_EXPORTER_ENDPOINT": "http://localhost:4318/v1/logs",
    "MEMORY_PRUNING_ENABLED": False,
    "MEMORY_PRUNING_L2_ENABLED": True,
    "MEMORY_PRUNING_L1_MUS_ENABLED": False,
    "MEMORY_PRUNING_L2_MUS_ENABLED": False,
    "MAX_PROJECT_MEMBERS": 3,
    "DEFAULT_MAX_SIMULATION_STEPS": 50,
    "MAX_KB_ENTRIES_FOR_PERCEPTION": 10,
    "MAX_KB_ENTRIES": 100,
    "MEMORY_STORE_TTL_SECONDS": 60 * 60 * 24 * 7,
    "MEMORY_STORE_PRUNE_INTERVAL_STEPS": 1,
    "MAX_IP_PER_TICK": 10.0,
    "MAX_DU_PER_TICK": 10.0,
    "GAS_PRICE_PER_CALL": 1.0,
    "GAS_PRICE_PER_TOKEN": 0.0,
    "SNAPSHOT_COMPRESS": False,
    "S3_BUCKET": "",
    "S3_PREFIX": "",
    "SNAPSHOT_INTERVAL_STEPS": 100,
    "MAX_AGENT_AGE": 10,
    "AGENT_TOKEN_BUDGET": 10000,
    "GENE_MUTATION_RATE": 0.1,
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
    "GAS_PRICE_PER_CALL",
    "GAS_PRICE_PER_TOKEN",
    "MAP_MOVE_IP_COST",
    "MAP_MOVE_IP_REWARD",
    "MAP_MOVE_DU_COST",
    "MAP_MOVE_DU_REWARD",
    "MAP_GATHER_IP_COST",
    "MAP_GATHER_IP_REWARD",
    "MAP_GATHER_DU_COST",
    "MAP_GATHER_DU_REWARD",
    "MAP_BUILD_IP_COST",
    "MAP_BUILD_IP_REWARD",
    "MAP_BUILD_DU_COST",
    "MAP_BUILD_DU_REWARD",
    "GENE_MUTATION_RATE",
]
INT_CONFIG_KEYS = [
    "IP_AWARD_FOR_PROPOSAL",
    "IP_COST_TO_POST_IDEA",
    "DU_AWARD_FOR_PROPOSAL",
    "LAW_PASS_IP_REWARD",
    "LAW_PASS_DU_REWARD",
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
    "MAX_KB_ENTRIES",
    "MEMORY_STORE_TTL_SECONDS",
    "MEMORY_STORE_PRUNE_INTERVAL_STEPS",
    "MEMORY_PRUNING_USAGE_COUNT_THRESHOLD",
    "SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS",
    "MAX_AGENT_AGE",
    "AGENT_TOKEN_BUDGET",
]
BOOL_CONFIG_KEYS = [
    "MEMORY_PRUNING_ENABLED",
    "MEMORY_PRUNING_L2_ENABLED",
    "MEMORY_PRUNING_L1_MUS_ENABLED",
    "MEMORY_PRUNING_L2_MUS_ENABLED",
    "SNAPSHOT_COMPRESS",
]

# Keys that must be defined for a complete runtime configuration.
# ``REDPANDA_BROKER`` enables event logging through Redpanda, while
# ``OPA_URL`` points to the Open Policy Agent service used to filter
# outgoing messages.
REQUIRED_CONFIG_KEYS = ["OLLAMA_API_BASE", "REDPANDA_BROKER", "MODEL_NAME", "OPA_URL"]



def load_config(*, validate_required: bool = True) -> dict[str, Any]:
    """Reload configuration from environment variables."""
    global settings, _CONFIG
    new_settings = ConfigSettings()
    try:
        data = new_settings.model_dump()
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        data = new_settings.dict()

    if validate_required:
        missing = [k for k in REQUIRED_CONFIG_KEYS if not data.get(k)]
        if missing:
            raise RuntimeError("Missing mandatory configuration keys: " + ", ".join(missing))

    settings = new_settings
    _CONFIG = data
    return cast(dict[str, Any], data)



def get_config(key: str | None = None) -> Any:
    """Return a configuration value from :class:`ConfigSettings`."""
    if key is None:
        try:
            return settings.model_dump()
        except AttributeError:  # pragma: no cover - pydantic v1
            return settings.dict()
    return getattr(settings, key)


def get(setting_name: str, default: str | None = None) -> object:
    """Retrieve a setting value with a fallback default."""
    return getattr(settings, setting_name, default)


RELATIONSHIP_LABELS = {
    (-1.0, -0.7): "Hostile",
    (-0.7, -0.4): "Negative",
    (-0.4, -0.1): "Cautious",
    (-0.1, 0.1): "Neutral",
    (0.1, 0.4): "Cordial",
    (0.4, 0.7): "Positive",
    (0.7, 1.0): "Allied",
}


def get_relationship_label(score: float) -> str:
    """Return a descriptive relationship label for ``score``."""
    for (min_val, max_val), label in RELATIONSHIP_LABELS.items():
        if min_val <= score <= max_val:
            return label
    return "Neutral"


def get_redis_config() -> dict[str, object]:
    """Return Redis connection details as a dictionary."""
    return {
        "host": settings.REDIS_HOST,
        "port": settings.REDIS_PORT,
        "db": settings.REDIS_DB,
        "password": getattr(settings, "REDIS_PASSWORD", None),
    }


def get_config_value_with_override(
    key: str, default: Any = None, module_name: str = "src.infra.config"
) -> Any:
    """Fetch a config value, checking overrides first."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        logger.error("Failed to import configuration module: %s", module_name)
        return default

    if CONFIG_OVERRIDES.get(key) is not None:
        return CONFIG_OVERRIDES[key]

    return getattr(module, str(key), default)


def __getattr__(name: str) -> Any:
    return getattr(settings, name)
