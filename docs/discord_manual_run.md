# Running SimulationDiscordBot Manually

This guide explains how to start the Discord bot and interact with it directly.

## Prerequisites
- Python dependencies installed (`pip install -r requirements.txt`)
- A running [Ollama](https://ollama.ai/) instance with the required model pulled
- A `.env` file configured with your Discord credentials:
  - `DISCORD_BOT_TOKEN`
  - `DISCORD_CHANNEL_ID`

## Starting the Bot
Launch the simulation with Discord integration:
```bash
python -m src.app --discord --steps 3
```
The bot connects to the channel specified by `DISCORD_CHANNEL_ID` and begins the simulation.

## Basic Commands
While the bot is running you can issue the following commands in Discord:
```text
!say hello world
```
Replies with:
```text
Simulated message received: hello world
```
```text
!stats
```
Displays runtime statistics similar to:
```text
LLM latency: 0 ms; KB size: 0
```
These commands are helpful for manual smoke testing of the Discord interface.

### Using Multiple Bot Tokens
You can run the simulation with several Discord bot accounts. Store the tokens
in a PostgreSQL table named `discord_tokens` with columns `agent_id` and `token`.
Set `DISCORD_TOKENS_DB_URL` to the database connection URL. When present,
`DISCORD_BOT_TOKEN` can be left blank or contain a comma-separated fallback
list.

Initialize the table:

```sql
-- scripts/init_discord_tokens.sql
CREATE TABLE IF NOT EXISTS discord_tokens (
    agent_id TEXT PRIMARY KEY,
    token TEXT NOT NULL
);
```

Load the SQL with `psql $DISCORD_TOKENS_DB_URL < scripts/init_discord_tokens.sql`.
