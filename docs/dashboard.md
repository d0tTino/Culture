# Dashboard API

## Misbehavior Feed

The dashboard backend exposes `/api/misbehavior` to retrieve recent misbehavior events recorded during simulation runs.

### Endpoint

`GET /api/misbehavior?limit=20`

- `limit` *(optional)*: maximum number of events to return. Defaults to 20.

### Response


```json
{
  "events": [
    {
      "step": 42,
      "agent_id": "abc123",
      "reason": "unauthorized action",
      "replay_path": "replay_42_42.jsonl"
    }
  ]
}
```

Each object contains the simulation `step`, the offending `agent_id`, the `reason` for the misbehavior, and a `replay_path` pointing to a replay slice containing events around that step.

## Agent Stats

`GET /api/agent_stats`

Returns mood, retrieval counts, and cost metrics for each agent.

### Response

```json
{
  "agents": {
    "abc123": {
      "mood": 0.5,
      "retrieval_count": 10,
      "du_per_1k_tokens": 1.23,
      "llm_latency_p95_ms": 450.0
    }
  }
}
```

- `du_per_1k_tokens`: DU cost per 1,000 generated tokens for the agent.
- `llm_latency_p95_ms`: 95th percentile latency of the agent's recent LLM calls in milliseconds.


