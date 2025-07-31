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

## 3. View semantic summaries

Select an agent ID to load its latest semantic summaries. The UI calls `/api/agents/{agent_id}/semantic_summaries` and lists the results below the controls.
