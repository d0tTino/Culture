# Observability Setup

This guide explains how to configure basic monitoring for Culture.ai using Grafana.

## 1. Requirements

- A running Prometheus instance collecting metrics from Culture.ai services
- Grafana installed and able to connect to Prometheus

## 2. Import the Dashboard

1. Open Grafana and navigate to **Dashboards > Import**.
2. Upload `docs/grafana_dashboard.json` from this repository.
3. Select your Prometheus data source when prompted and click **Import**.

The dashboard provides a starter panel showing CPU usage for Culture.ai processes. Extend it with additional panels as needed.

## 3. Running Grafana Locally

If you want to run Grafana locally for quick testing, you can use Docker:

```bash
docker run -d -p 3000:3000 grafana/grafana
```

Once running, access Grafana at [http://localhost:3000](http://localhost:3000) and follow the import steps above.

