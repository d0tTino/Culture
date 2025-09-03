# Dashboard

## Misbehavior feed

The dashboard backend exposes a `GET /api/misbehavior` endpoint for auditing
misbehavior events. It accepts an optional `limit` query parameter (default
`20`) to cap the number of returned records; non-positive values yield an empty
list. Each entry contains:

- `step`: Simulation tick when the event occurred.
- `detail`: Description of the misbehavior.
- `replay`: Filename of a replay slice covering the step.

Example response:

```json
{
  "events": [
    {
      "step": 42,
      "detail": "unexpected action",
      "replay": "replay_42_42.jsonl"
    }
  ]
}
```
