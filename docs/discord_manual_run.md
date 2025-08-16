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

### Slash Commands

When channels are mapped to specific agents you can also use slash commands:

```text
/start
```
Starts the simulation if it is currently paused.

```text
/stop
```
Stops the simulation.

```text
/spawn agent_4
```
Spawns a new agent with the given ID.

```text
/status
```
Shows the agent's current IP and DU balance as an ephemeral message.

```text
/stats
```
Displays latency and Knowledge Board size, also ephemeral. These commands only
respond if the mapped agent has remaining IP and DU.

```text
/broadcast Hello from the outside
```
Sends a broadcast message to all agents.

```text
/kb Add multi-agent architecture diagram to the KB
```
Adds a new entry to the shared Knowledge Board.

### Moderation Commands

The bot exposes moderation-oriented commands:

```text
/mute agent_4
```
Mutes an agent until it is explicitly unmuted.

```text
/unmute agent_4
```
Restores the agent's ability to speak.

```text
/reset_memory agent_4
```
Clears an agent's memory.

```text
/penalty agent_4 ip:1 du:1
```
Applies an IP/DU penalty to the agent.

All moderation commands are rate limited on a per-agent basis. Exceeding the
limit automatically issues IP and DU penalties as a safety rail and records the
deduction in the ledger. For example:

```text
/mute agent_4
/mute agent_4
```
The second call returns:

```text
rate limited
```
and `agent_4` loses IP and DU according to the configured penalty values.

### Administrative Commands

```text
/set_max_rate 10
```
Sets the maximum number of commands a user may invoke before being throttled.
Requires administrator permissions.

```text
/kill
```
Shuts down the bot. Administrator only.

### Using Multiple Bot Tokens
You can run the simulation with several Discord bot accounts. Tokens are stored
in a PostgreSQL table named `discord_tokens` with columns `agent_id` and
`token`. Set `DISCORD_TOKENS_DB_URL` to the database connection URL. When this
value is provided the application will create the table automatically using
SQLAlchemy. `DISCORD_BOT_TOKEN` can be left blank or contain a comma-separated
fallback list.

### Troubleshooting Permission Errors
If slash commands fail or the bot cannot send messages:
- Ensure the bot role has **Send Messages**, **Read Message History**, and **Use
  Application Commands** permissions in the target channel.
- Double-check that `DISCORD_CHANNEL_ID` points to a channel the bot can
  access.
- Re-invite the bot with the `applications.commands` scope if commands do not
  appear.
