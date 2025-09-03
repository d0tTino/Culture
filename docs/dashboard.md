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


