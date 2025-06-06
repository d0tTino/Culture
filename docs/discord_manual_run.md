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
