# Observability Setup

This guide explains how to configure basic monitoring for Culture.ai using Grafana.

## 1. Requirements

- A running Prometheus instance collecting metrics from Culture.ai services
- Grafana installed and able to connect to Prometheus

## 2. Import the Dashboard

1. Open Grafana and navigate to **Dashboards > Import**.
2. Upload `docs/grafana_dashboard.json` from this repository.
3. Select your Prometheus data source when prompted and click **Import**.

The imported dashboard includes panels for CPU usage, Knowledge Board size, active agent count, and the LLM query rate (QPS).

## 3. Running Grafana Locally

If you want to run Grafana locally for quick testing, you can use Docker:

```bash
docker run -d -p 3000:3000 grafana/grafana
```

Once running, access Grafana at [http://localhost:3000](http://localhost:3000) and follow the import steps above.

## 4. OpenTelemetry Logs

Culture.ai can export structured logs via the OpenTelemetry OTLP exporter. The exporter
sends logs to `localhost:4318/v1/logs` by default. Set `OTEL_EXPORTER_ENDPOINT` to
override this URL, and set `ENABLE_OTEL=1` in your `.env` file to activate the exporter.

To receive these logs locally, run an OTLP-compatible collector such as the
[OpenTelemetry Collector](https://opentelemetry.io/docs/collector/):

```bash
otelcol --config=your_config.yaml
```

You should then see logs arriving on port `4318`.

## 5. Debugging SQLite Locks

If you encounter database lock errors during development, enable SQLite debug mode:

```bash
export DEBUG_SQLITE=1
```

This sets the database to WAL mode and increases the busy timeout to help diagnose locking issues.

