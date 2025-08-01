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

The imported dashboard includes panels for CPU usage, Knowledge Board size, active agent count, and the LLM query rate (QPS).

To monitor memory retrievals, add a new panel in Grafana using the `memory_retrievals_total` and `memory_retrieval_errors_total` counters. For example, graph `rate(memory_retrievals_total[1m])` to see retrieval throughput.

Additional Prometheus metrics include:

- `llm_errors_total` – counts failed LLM calls captured by the monitoring decorator.
- `memory_retrievals_total` – counts successful memory lookups.
- `memory_retrieval_errors_total` – counts memory retrieval failures.

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
```

Events will be written to the `culture-events` topic. You can inspect them with the `rpk` CLI or any Kafka-compatible consumer.

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

