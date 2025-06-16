-- Initialize discord_tokens table
CREATE TABLE IF NOT EXISTS discord_tokens (
    agent_id TEXT PRIMARY KEY,
    token TEXT NOT NULL
);
