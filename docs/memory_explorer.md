# Memory Explorer

This guide explains how to inspect an agent's semantic summaries using the Memory Explorer page.

## 1. Start the dashboard backend

Launch the HTTP service so the UI can fetch data:

```bash
python -m src.http_app
```

By default the server listens on `http://localhost:8000`.

## 2. Open the Memory Explorer page

With the backend running, navigate to `/memory` in your browser. When using the development server this is usually `http://localhost:5173/memory`.

## 3. View agent data

Select an agent ID to load its latest information:

- `/api/agents/{agent_id}/state` returns the agent's current state such as resource levels.
- `/api/agents/{agent_id}/memories` returns recent raw memory entries.
- `/api/agents/{agent_id}/semantic_summaries` provides high-level summaries.

The Memory Explorer lists these results below the controls so you can inspect how an agent is evolving over time.
