"""Application configuration settings using pydantic."""

from __future__ import annotations

from pydantic_settings import BaseSettings

BaseSettings = _BaseSettings  # type: ignore[misc]

class ConfigSettings(BaseSettings):
    """Configuration loaded from environment variables and ``.env`` file."""

    OLLAMA_API_BASE: str = "http://localhost:11434"
    MODEL_NAME: str = ""
    ROLE_DU_GENERATION: dict[str, float] = {
        "Facilitator": 1.0,
        "Innovator": 1.0,
        "Analyzer": 1.0,
    }

    DEFAULT_LLM_MODEL: str = "mistral:latest"
    # Backwards compatibility with older config keys
    MODEL_NAME: str = "mistral:latest"
    DEFAULT_TEMPERATURE: float = 0.7
    MEMORY_THRESHOLD_L1: float = 0.2
    MEMORY_THRESHOLD_L2: float = 0.3
    VECTOR_STORE_DIR: str = "./chroma_db"
    VECTOR_STORE_BACKEND: str = "chroma"
    KNOWLEDGE_BOARD_BACKEND: str = "memory"
    GRAPH_DB_URI: str = "bolt://localhost:7687"
    GRAPH_DB_USER: str = "neo4j"
    GRAPH_DB_PASSWORD: str = "test"
    WEAVIATE_URL: str = "http://localhost:8080"
    OLLAMA_REQUEST_TIMEOUT: int = 10
    REDPANDA_BROKER: str = ""
    OPA_URL: str = ""
    MODEL_NAME: str = "mistral:latest"
    ROLE_DU_GENERATION: dict[str, float] = {}
    REDPANDA_TOPIC: str = "culture.events"
    REPLAY_GROUP: str = "culture-replay"
    ENABLE_REDPANDA: bool = False
    TARGETED_MESSAGE_MULTIPLIER: float = 3.0
    POSITIVE_RELATIONSHIP_LEARNING_RATE: float = 0.3
    NEGATIVE_RELATIONSHIP_LEARNING_RATE: float = 0.4
    RELATIONSHIP_DECAY_FACTOR: float = 0.01
    MOOD_DECAY_FACTOR: float = 0.02
    ROLE_CHANGE_IP_COST: float = 5.0
    NEUTRAL_RELATIONSHIP_LEARNING_RATE: float = 0.1
    INITIAL_INFLUENCE_POINTS: float = 10.0
    INITIAL_DATA_UNITS: float = 20.0
    MAX_SHORT_TERM_MEMORY: int = 10
    SHORT_TERM_MEMORY_DECAY_RATE: float = 0.1
    MIN_RELATIONSHIP_SCORE: float = -1.0
    MAX_RELATIONSHIP_SCORE: float = 1.0
    MOOD_UPDATE_RATE: float = 0.2
    IP_COST_SEND_DIRECT_MESSAGE: float = 1.0
    DU_COST_PER_ACTION: float = 1.0
    ROLE_CHANGE_COOLDOWN: int = 3
    IP_AWARD_FOR_PROPOSAL: int = 5
    IP_COST_TO_POST_IDEA: int = 2
    DU_AWARD_FOR_PROPOSAL: int = 1
    LAW_PASS_IP_REWARD: int = 1
    LAW_PASS_DU_REWARD: int = 0
    IP_COST_CREATE_PROJECT: int = 10
    IP_COST_JOIN_PROJECT: int = 1
    IP_AWARD_FACILITATION_ATTEMPT: int = 3
    PROPOSE_DETAILED_IDEA_DU_COST: int = 5
    DU_AWARD_IDEA_ACKNOWLEDGED: int = 3
    DU_AWARD_SUCCESSFUL_ANALYSIS: int = 4
    DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE: int = 1
    DU_COST_DEEP_ANALYSIS: int = 3
    DU_COST_REQUEST_DETAILED_CLARIFICATION: int = 2
    DU_COST_CREATE_PROJECT: int = 10
    DU_COST_JOIN_PROJECT: int = 1
    MAP_MOVE_IP_COST: float = 0.0
    MAP_MOVE_IP_REWARD: float = 0.0
    MAP_MOVE_DU_COST: float = 0.0
    MAP_MOVE_DU_REWARD: float = 0.0
    MAP_GATHER_IP_COST: float = 0.0
    MAP_GATHER_IP_REWARD: float = 0.0
    MAP_GATHER_DU_COST: float = 0.0
    MAP_GATHER_DU_REWARD: float = 0.0
    MAP_BUILD_IP_COST: float = 0.0
    MAP_BUILD_IP_REWARD: float = 0.0
    MAP_BUILD_DU_COST: float = 0.0
    MAP_BUILD_DU_REWARD: float = 0.0
    MEMORY_PRUNING_L1_DELAY_STEPS: int = 10
    MEMORY_PRUNING_L2_MAX_AGE_DAYS: int = 30
    MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS: int = 100
    MEMORY_PRUNING_L1_MUS_THRESHOLD: float = 0.3
    MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION: int = 7
    MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS: int = 50
    MEMORY_PRUNING_L2_MUS_THRESHOLD: float = 0.25
    MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION: int = 14
    MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS: int = 150
    MEMORY_PRUNING_USAGE_COUNT_THRESHOLD: int = 3
    SEMANTIC_MEMORY_CONSOLIDATION_INTERVAL_STEPS: int = 0
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    DISCORD_BOT_TOKEN: str = ""
    DISCORD_CHANNEL_ID: int | None = None
    DISCORD_TOKENS_DB_URL: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    DEFAULT_LOG_LEVEL: str = "INFO"
    OTEL_EXPORTER_ENDPOINT: str = "http://localhost:4318/v1/logs"
    MEMORY_PRUNING_ENABLED: bool = False
    MEMORY_PRUNING_L2_ENABLED: bool = True
    MEMORY_PRUNING_L1_MUS_ENABLED: bool = False
    MEMORY_PRUNING_L2_MUS_ENABLED: bool = False
    MAX_PROJECT_MEMBERS: int = 3
    DEFAULT_MAX_SIMULATION_STEPS: int = 50
    MAX_KB_ENTRIES_FOR_PERCEPTION: int = 10
    MAX_KB_ENTRIES: int = 100
    MEMORY_STORE_TTL_SECONDS: int = 60 * 60 * 24 * 7
    MEMORY_STORE_PRUNE_INTERVAL_STEPS: int = 1
    MAX_IP_PER_TICK: float = 10.0
    MAX_DU_PER_TICK: float = 10.0
    GAS_PRICE_PER_CALL: float = 1.0
    GAS_PRICE_PER_TOKEN: float = 0.0
    SNAPSHOT_COMPRESS: bool = False
    S3_BUCKET: str = ""
    S3_PREFIX: str = ""
    SNAPSHOT_INTERVAL_STEPS: int = 100
    MAX_AGENT_AGE: int = 10
    AGENT_TOKEN_BUDGET: int = 10000
    ROLE_DU_GENERATION: dict[str, object] = {
        "Facilitator": {"base": 1.0},
        "Innovator": {"base": 1.0},
        "Analyzer": {"base": 1.0},
    }
    GENE_MUTATION_RATE: float = 0.1
    ENABLE_OTEL: bool = False
    OPA_BLOCKLIST: str = ""
    HTTP_HOST: str = "0.0.0.0"
    HTTP_PORT: int = 8000
    DEBUG_SQLITE: bool = False
    ENABLE_DSPY_TESTS: bool = False


    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = ConfigSettings()
