# Observability Setup

This guide explains how to configure basic monitoring for Culture.ai using Grafana.

## 1. Requirements

- A running Prometheus instance collecting metrics from Culture.ai services
- Grafana installed and able to connect to Prometheus

## 2. Running Prometheus Locally

Create a file named `prometheus.yml` with the following contents:

```yaml
global:
  scrape_interval: 10s
scrape_configs:
  - job_name: "culture"
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

Start Prometheus with Docker:

```bash
docker run -d -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

Visit [http://localhost:9090](http://localhost:9090) to confirm metrics are being scraped.

## 3. Import the Dashboard

1. Open Grafana and navigate to **Dashboards > Import**.
2. Upload `docs/grafana_dashboard.json` from this repository.
3. Select your Prometheus data source when prompted and click **Import**.
4. Or import via the API:

```bash
curl -X POST -H "Content-Type: application/json" \
    -d @docs/grafana_dashboard.json \
    http://admin:admin@localhost:3000/api/dashboards/db
```

The imported dashboard includes time-series panels for coalition count, average sentiment, proposal throughput, DU per 1k tokens, LLM p95 latency, memory retrieval throughput/error rate, and human message volume. These views are wired to Prometheus so you can import the JSON and begin charting the metrics immediately.

### Panel Queries

Each panel relies on the following Prometheus expressions:

- **DU per 1k Tokens:** `llm_du_per_1k_tokens`
- **LLM p95 Latency:** `llm_latency_p95_ms`
- **Memory Retrieval Throughput:** `rate(memory_retrievals_total[5m])`
- **Memory Retrieval Error Rate:** `rate(memory_retrieval_errors_total[5m])`
- **Human Message Volume:** `rate(human_messages_total[5m])`

### Memory Retrieval Metrics

Two counters track memory lookup activity:

- `memory_retrievals_total` – counts successful memory lookups.
- `memory_retrieval_errors_total` – counts memory retrieval failures.

The dashboard's **Memory Retrieval Throughput / Error Rate** panel already charts these counters. It graphs `rate(memory_retrievals_total[5m])` for successful lookups and `rate(memory_retrieval_errors_total[5m])` for failures so you can compare throughput against errors in a single view.

Additional Prometheus metrics include:

- `llm_errors_total` – counts failed LLM calls captured by the monitoring decorator.
- `memory_retrievals_total` – counts successful memory lookups.
- `memory_retrieval_errors_total` – counts memory retrieval failures.

### Monitoring DU Budgets

Each agent exposes its remaining DU balance and efficiency via Prometheus gauges:

- `agent_remaining_du{agent_id="<id>"}` shows the DU balance for a given agent.
- `agent_du_per_1k_tokens{agent_id="<id>"}` reports DU spent per 1k tokens on the last LLM call.

The **DU per 1k Tokens** panel visualizes `llm_du_per_1k_tokens` to highlight overall efficiency. Create additional views with `avg by (agent_id) (agent_du_per_1k_tokens)` or `agent_remaining_du` if you need to track per-agent budgets.

### LLM Tail Latency

Use the **LLM p95 Latency** panel to monitor tail behavior. It charts the `llm_latency_p95_ms` gauge exposed by the dashboard backend. For per-agent tail latency, graph `agent_llm_latency_p95_ms` or aggregate with `max by (agent_id) (agent_llm_latency_p95_ms)`.

### Human Message Volume

The **Human Message Volume** panel plots `rate(human_messages_total[5m])` to show how frequently humans interact with the simulation. Adjust the range selector (for example `rate(human_messages_total[1m])`) to zoom in on shorter bursts of activity.

### Dashboard Cost Metrics Endpoint

The dashboard backend provides `/api/observability_metrics` for quick visibility into
LLM usage, coalition dynamics, and retrieval health. Example response:

```json
{
  "du_per_1k_tokens": 1.8,
  "llm_latency_p95_ms": 450.0,
  "coalition_count": 4,
  "average_sentiment": 0.37,
  "rag_hit_rate": 0.82,
  "memory_retrievals_total": 120,
  "memory_retrieval_errors_total": 6,
  "memory_retrieval_success_rate": 0.95,
  "memory_retrieval_error_rate": 0.05,
  "llm_errors_total": 3,
  "llm_error_rate": 0.02
}
```

- `du_per_1k_tokens` – average digital units spent per 1,000 tokens. Lower values
  indicate more efficient usage of the DU budget.
- `llm_latency_p95_ms` – 95th percentile latency of recent LLM calls in milliseconds,
  useful for spotting tail latency issues.
- `coalition_count` – number of active coalitions discovered in the simulation.
- `average_sentiment` – current aggregate sentiment across agents.
- `rag_hit_rate` – most recent hit rate for Retrieval Augmented Generation (RAG)
  lookups.
- `memory_retrievals_total` and `memory_retrieval_errors_total` – cumulative counts of
  successful and failed memory retrievals, respectively.
- `memory_retrieval_success_rate` and `memory_retrieval_error_rate` – derived ratios of
  successful and failed retrievals.
- `llm_errors_total` – total failed LLM calls captured by the monitoring decorator.
- `llm_error_rate` – share of failed LLM calls relative to total LLM traffic.

These metrics can be fetched directly by the UI or external monitoring systems.

## 4. Running Grafana Locally

If you want to run Grafana locally for quick testing, you can use Docker. The default login is `admin`/`admin`:

```bash
docker run -d -p 3000:3000 grafana/grafana
```

Once running, access Grafana at [http://localhost:3000](http://localhost:3000) and follow the import steps above.

## 5. OpenTelemetry Logs

Culture.ai can export structured logs via the OpenTelemetry OTLP exporter. The exporter
sends logs to `localhost:4318/v1/logs` by default. Copy `.env.example` to `.env` and
set the following variables to enable exporting:

```env
ENABLE_OTEL=1
OTEL_EXPORTER_ENDPOINT=http://localhost:4318/v1/logs
```
Adjust the endpoint if your collector runs elsewhere.

To receive these logs locally, run an OTLP-compatible collector such as the
[OpenTelemetry Collector](https://opentelemetry.io/docs/collector/):

```bash
otelcol --config=your_config.yaml
```

You should then see logs arriving on port `4318`.

### Tracing

The simulation also emits OpenTelemetry spans for deeper insight into runtime
behavior:

- `memory.retrieve`, `memory.episodic_retrieve`, and `memory.semantic_retrieve`
  capture memory lookup paths and latency.
- `llm.request` and `llm.du_burn` record LLM API calls and digital unit charges.
- `discord.command` and `discord.send_message` trace Discord command handlers
  and outbound messages. Moderation slash commands (mute, unmute, penalty,
  reset_memory) now emit their own `discord.command` spans so approvals and
  rate-limit violations are easy to inspect in tracing tools.

To view traces locally, run a collector such as
[Jaeger](https://www.jaegertracing.io/) and set:

```env
ENABLE_OTEL=1
OTEL_EXPORTER_ENDPOINT=http://localhost:4318/v1/traces
```

Then open Jaeger's UI (default `http://localhost:16686`) to explore the spans.

## 6. Debugging SQLite Locks

If you encounter database lock errors during development, enable SQLite debug mode:

```bash
export DEBUG_SQLITE=1
```

This sets the database to WAL mode and increases the busy timeout to help diagnose locking issues.

## 7. Policy Engine (OPA)

Culture.ai can optionally send outgoing messages through an [Open Policy Agent](https://www.openpolicyagent.org/) service for additional filtering. Set the `OPA_URL` environment variable to point at your OPA policy endpoint (for example `http://localhost:8181/v1/data/discord/allow`). The endpoint should return JSON in the form:

```json
{
  "result": {"allow": true, "content": "optional modified text"}
}
```

If `allow` is `false`, the message will be blocked. If `content` is returned, it will replace the original text before sending.

## 8. Redpanda Event Log

Culture.ai can stream all simulation events to a Redpanda broker for later replay and analysis.
Follow [docs/redpanda_setup.md](redpanda_setup.md) to install Redpanda locally and start it with Docker Compose.
Set the following variables in your `.env`:

```env
ENABLE_REDPANDA=1
REDPANDA_BROKER=localhost:9092
REDPANDA_TOPIC=culture.events  # optional override of the topic name
```

Events will be written to the `culture.events` topic by default. You can inspect them with the `rpk` CLI or any Kafka-compatible consumer, or override the destination by setting `REDPANDA_TOPIC`.

## 9. Metrics Reference

Prometheus metrics exported by the simulation include:

- `active_agent_count` – number of agents currently active
- `llm_calls_total` – total LLM invocations
- `llm_latency_ms` – latency of the last LLM call
- `llm_errors_total` – failed LLM calls
- `knowledge_board_size` – total Knowledge Board entries
- `event_bus_queue_size` – number of queues subscribed to the event bus
- `memory_retrievals_total` – successful memory retrievals
- `memory_retrieval_errors_total` – failed memory retrievals

