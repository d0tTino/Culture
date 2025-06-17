# Redpanda Setup

This guide explains how to install and run Redpanda for event logging and deterministic replay in Culture.ai.

## Install Redpanda

The easiest way to install Redpanda on Linux is via the official install script:

```bash
curl -1s https://raw.githubusercontent.com/redpanda-data/redpanda/master/install.sh | bash
```

This script adds the package repository and installs the latest stable Redpanda release.

## Docker Compose Example

You can also run Redpanda using Docker Compose. Save the following as `docker-compose.redpanda.yml`:

```yaml
version: '3.8'
services:
  redpanda:
    image: docker.redpanda.com/redpandadata/redpanda:latest
    command: >
      redpanda start --overprovisioned --smp 1 \
      --memory 1G --reserve-memory 0M --node-id 0
    ports:
      - "9092:9092"
      - "9644:9644"
    volumes:
      - redpanda_data:/var/lib/redpanda/data
volumes:
  redpanda_data:
```

Start the service:

```bash
docker compose -f docker-compose.redpanda.yml up -d
```

## Environment Variables

Set the following variables in your `.env` file so Culture.ai can connect:

```env
ENABLE_REDPANDA=1
REDPANDA_BROKER=localhost:9092
```

`ENABLE_REDPANDA` enables event logging and replay. `REDPANDA_BROKER` should point to the broker address used by your Docker Compose setup.
