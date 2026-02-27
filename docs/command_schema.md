# Canonical Simulation Command Schema (v1)

`SimulationCommandService` consumes a canonical envelope for all command mutations.
All transports (Discord, dashboard, tests, internal code, and future web UI) should adapt into this shape.

## Versioning

- **Current schema version**: `1`
- Recommended transport payload metadata:
  - `schema_version`: integer (`1`)
  - `command_type`: canonical intent
- Backward-compatible aliases remain accepted by `parse_bus_command`.

## Canonical envelope

```json
{
  "intent": "human_message|direct_message|broadcast|knowledge_board|spawn|moderation|control|inject_event",
  "content": "optional string",
  "action": "optional command action (e.g. set_speed)",
  "value": 1.0,
  "tags": ["optional", "tags"],
  "prompt": "optional prompt/scope",
  "text": "optional freeform text",
  "agent_id": "optional target/author agent",
  "role": "optional role string or object",
  "persona": "optional persona",
  "backstory": "optional backstory",
  "traits": {"openness": 0.5},
  "routing": {
    "sender_id": "human",
    "source": "discord|dashboard|internal",
    "channel_id": "optional",
    "recipient_id": "optional",
    "target_agent_id": "optional"
  },
  "auth": {"permissions": ["admin", "moderator"]},
  "budget": {"budget_agent_id": "optional", "attribution_scope": "default"},
  "metadata": {"raw": "payload"}
}
```

## Alias compatibility (v1)

The command bus normalizer accepts legacy aliases and maps them into the canonical envelope:

- `command=dm` → `intent=direct_message`
- `command=kb` → `intent=knowledge_board`
- `command in {pause,resume,pause_all,start,stop,set_speed,kill_agent}` → `intent=control`
- `command=inject_event` → `intent=inject_event`
- `author` → `agent_id`
- `scope` → `prompt`
- `text` fallbacks from `content`

This allows older Discord/dashboard/test payloads to work while transports migrate to explicit canonical fields.
