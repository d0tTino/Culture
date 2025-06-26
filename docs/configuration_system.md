# Culture.ai Configuration System

This document explains how to configure the Culture.ai simulation using the centralized configuration system.

## Overview

The configuration system in Culture.ai allows you to customize various aspects of the simulation without modifying code. Settings are loaded from the following sources (in order of precedence):

1. Environment variables
2. `.env` file in the project root
3. Default values defined in `src/infra/config.py`

## Getting Started

1. Create a `.env` file in the project root by copying the `.env.example` file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file to customize the settings you want to change.

3. Run the simulation. The settings will be automatically loaded.

## Configuration Categories

The configuration is organized into the following categories:

### Basic Configuration

- `LOG_LEVEL` - Logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)

### API Keys

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key

### LLM Settings

- `DEFAULT_MODEL` - Default LLM model to use
- `DEFAULT_TEMPERATURE` - Temperature setting for LLM responses

### Ollama Settings

- `OLLAMA_API_BASE` - Base URL for Ollama API
- `OLLAMA_REQUEST_TIMEOUT` - Timeout in seconds for Ollama requests

### Redis Settings

- `REDIS_HOST` - Redis hostname
- `REDIS_PORT` - Redis port
- `REDIS_DB` - Redis database number
- `REDIS_PASSWORD` - Redis password

### Discord Bot Settings

- `DISCORD_BOT_TOKEN` - Discord bot token or comma-separated tokens
- `DISCORD_CHANNEL_ID` - Discord channel ID
- `DISCORD_TOKENS_DB_URL` - PostgreSQL URL for the `discord_tokens` table. The
  table is created automatically via SQLAlchemy when this value is set.

### Memory Pruning Settings

Regular memory pruning settings:
- `MEMORY_PRUNING_ENABLED` - Whether to enable memory pruning
- `MEMORY_PRUNING_L1_DELAY_STEPS` - Steps to wait after L2 summary before pruning L1 summaries
- `MEMORY_PRUNING_L2_ENABLED` - Whether to enable L2 summary pruning
- `MEMORY_PRUNING_L2_MAX_AGE_DAYS` - Max age for L2 summaries
- `MEMORY_PRUNING_L2_CHECK_INTERVAL_STEPS` - Steps between L2 pruning checks

### MUS-based Memory Pruning Settings

L1 MUS-based pruning:
- `MEMORY_PRUNING_L1_MUS_ENABLED` - Whether to enable L1 MUS-based pruning
- `MEMORY_PRUNING_L1_MUS_THRESHOLD` - Threshold for L1 MUS pruning
- `MEMORY_PRUNING_L1_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION` - Minimum age for L1 memories to be considered
- `MEMORY_PRUNING_L1_MUS_CHECK_INTERVAL_STEPS` - Steps between L1 MUS checks

L2 MUS-based pruning:
- `MEMORY_PRUNING_L2_MUS_ENABLED` - Whether to enable L2 MUS-based pruning
- `MEMORY_PRUNING_L2_MUS_THRESHOLD` - Threshold for L2 MUS pruning
- `MEMORY_PRUNING_L2_MUS_MIN_AGE_DAYS_FOR_CONSIDERATION` - Minimum age for L2 memories to be considered
- `MEMORY_PRUNING_L2_MUS_CHECK_INTERVAL_STEPS` - Steps between L2 MUS checks

### Mood and Relationship Settings

- `MOOD_DECAY_FACTOR` - Rate at which agent mood decays toward neutral
- `RELATIONSHIP_DECAY_FACTOR` - Rate at which relationships decay toward neutral
- `POSITIVE_RELATIONSHIP_LEARNING_RATE` - Learning rate for positive interactions
- `NEGATIVE_RELATIONSHIP_LEARNING_RATE` - Learning rate for negative interactions
- `NEUTRAL_RELATIONSHIP_LEARNING_RATE` - Learning rate for neutral interactions
- `TARGETED_MESSAGE_MULTIPLIER` - Multiplier for relationship changes from targeted messages

### Influence Points (IP) Settings

- `INITIAL_INFLUENCE_POINTS` - Starting IP for new agents
- `IP_AWARD_FOR_PROPOSAL` - IP awarded for successfully proposing an idea
- `IP_COST_TO_POST_IDEA` - IP cost to post an idea
- `ROLE_CHANGE_IP_COST` - IP cost to change roles
- `IP_COST_CREATE_PROJECT` - IP cost to create a new project
- `IP_COST_JOIN_PROJECT` - IP cost to join an existing project

### Data Units (DU) Settings

- `INITIAL_DATA_UNITS` - Starting DU for new agents
- `PROPOSE_DETAILED_IDEA_DU_COST` - DU cost for posting a detailed idea
- `DU_AWARD_IDEA_ACKNOWLEDGED` - DU awarded when an idea is acknowledged
- `DU_AWARD_SUCCESSFUL_ANALYSIS` - DU awarded for a successful analysis
- `DU_BONUS_FOR_CONSTRUCTIVE_REFERENCE` - DU bonus for constructive reference
- `DU_COST_DEEP_ANALYSIS` - DU cost for deep analysis
- `DU_COST_REQUEST_DETAILED_CLARIFICATION` - DU cost for detailed clarification
- `DU_COST_CREATE_PROJECT` - DU cost to create a new project
- `DU_COST_JOIN_PROJECT` - DU cost to join an existing project

### Role Settings

- `ROLE_DU_GENERATION_INNOVATOR` - DU generated per turn for Innovator role
- `ROLE_DU_GENERATION_ANALYZER` - DU generated per turn for Analyzer role
- `ROLE_DU_GENERATION_FACILITATOR` - DU generated per turn for Facilitator role
- `ROLE_DU_GENERATION_DEFAULT` - DU generated per turn for default contributor role
- `ROLE_CHANGE_COOLDOWN` - Minimum steps an agent must stay in a role

### Project Settings

- `MAX_PROJECT_MEMBERS` - Maximum number of members in a project

### Observability Settings

- `ENABLE_OTEL` - Set to `1` to enable OpenTelemetry log export
- `OTEL_EXPORTER_ENDPOINT` - Override the OTLP exporter URL
- `ENABLE_REDPANDA` - Set to `1` to enable Redpanda event logging
- `REDPANDA_BROKER` - Address of the Redpanda broker
- `SNAPSHOT_COMPRESS` - Set to `1` to compress simulation snapshots with zstd
- `S3_BUCKET` - Optional S3 bucket for snapshot storage
- `S3_PREFIX` - Prefix within the S3 bucket for snapshot files

## Configuration Access in Code

To access configuration values in your code, import the config module:

```python
from src.infra import config

# Access configuration values
model_name = config.DEFAULT_MODEL
cooldown = config.ROLE_CHANGE_COOLDOWN
```

For dynamic access by name, use the `get` method:

```python
from src.infra import config

# Access a configuration value by name
cooldown = config.get("ROLE_CHANGE_COOLDOWN")
```

## Adding New Configuration Parameters

To add a new configuration parameter:

1. Add it to `src/infra/config.py` with appropriate type conversion and default value
2. Add it to `.env.example` with documentation
3. Use it in your code by importing and referencing `config.YOUR_PARAMETER_NAME` 
