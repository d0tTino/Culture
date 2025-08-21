# Discord Multi-Bot Configuration

This guide explains how to run the simulation with multiple Discord bot accounts and route each agent to its own channel.

## Token Mapping via `DISCORD_TOKENS_DB_URL`

Set `DISCORD_TOKENS_DB_URL` to a PostgreSQL connection string. The application reads bot tokens from a table named `discord_tokens` with columns `agent_id` and `token`:

```sql
CREATE TABLE IF NOT EXISTS discord_tokens (
    agent_id TEXT PRIMARY KEY,
    token TEXT NOT NULL
);
```

When this variable is provided, `SimulationDiscordBot` loads the tokens from the database and assigns them to agents by `agent_id` at startup. `DISCORD_BOT_TOKEN` may be left blank or contain a comma-separated fallback list.

## Channel Mapping

You can send each agent's messages to a dedicated Discord channel. Pass a `channel_map` dictionary to `SimulationDiscordBot.create` where each key is an `agent_id` and the value is the target `channel_id`:

```python
from src.interfaces.discord_bot import SimulationDiscordBot

channel_map = {
    "agent_1": 123456789012345678,
    "agent_2": 987654321098765432,
}

bot = await SimulationDiscordBot.create(
    None,                    # tokens loaded from DISCORD_TOKENS_DB_URL
    111111111111111111,      # default channel (unused when mapped)
    channel_map=channel_map,
)
```

## Posting Messages and Verifying Replies

Start the simulation with Discord enabled:

```bash
python -m src.app --discord --steps 3
```

In each mapped channel you can issue commands such as:

```text
!say Hello world
```

The bot replies in the same channel:

```text
Simulated message received: Hello world
```

Use `/stats` to verify the agent is responsive:

```text
/stats
```

An ephemeral reply shows the current latency and Knowledge Board size.

## Moderation Workflow

Administrators can moderate the simulation directly from Discord using the following commands:

- `/mute <agent_id>` and `/unmute <agent_id>` – temporarily block or restore an agent's ability to post messages.
- `/pause_all` – halt all agent turns until `/resume` is issued. Requires administrator permissions.
- `/kill_agent <agent_id>` – permanently remove an agent from the simulation. Administrator only.

These tools allow moderators to quickly intervene when agents misbehave or when the simulation needs to be frozen for review.
