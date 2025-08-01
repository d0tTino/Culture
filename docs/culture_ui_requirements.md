# Culture UI Requirements

This document summarizes the current requirements for the Culture UI as tracked in the living requirements brief. The UI is a lightweight React + TypeScript application used to monitor simulations and inspect agent data.

## Functional Requirements

| ID | Requirement | Description |
| --- | ----------- | ----------- |
| UI-1 | Mission Overview | Provide a table of active missions with ID, name, status, and progress. Rows should be draggable to reorder by priority. |
| UI-2 | Agent Data Overview | Display a paginated table of agent observations and messages. |
| UI-3 | Live Event Stream | Connect to `/stream/events` via Server-Sent Events (SSE) and fall back to WebSocket when SSE is unavailable. |
| UI-4 | Widget System | Support a pluggable widget architecture so new panels can be registered dynamically. |
| UI-5 | Dark Mode | Allow users to toggle between light and dark themes. |

## Non‑Functional Requirements

| ID | Requirement | Description |
| --- | ----------- | ----------- |
| NF-1 | Responsive Design | The UI should scale to desktop and tablet screens. |
| NF-2 | Accessibility | Components must meet basic accessibility guidelines (ARIA labels, keyboard navigation). |
| NF-3 | Type Safety | All code is written in TypeScript and checked with `pnpm type-check`. |
| NF-4 | Linting | ESLint is run via `pnpm lint` in CI. |

## WidgetRegistry Interface

The widget system relies on a simple registry interface that allows pages to register custom widgets at runtime:

```ts
export interface WidgetRegistry {
  register(name: string, component: React.ComponentType): void
  get(name: string): React.ComponentType | undefined
  list(): string[]
}
```

Widgets are rendered based on the registry contents, enabling third‑party extensions without modifying core UI files.

## Timeline and Breakpoints

The dashboard includes a **Timeline** widget that visualizes simulation steps. Users can scrub through completed steps using a slider control.

Events may carry tags such as `violence`, `nsfw`, or `sabotage`. When a tag matches one of the configured breakpoints the simulation automatically pauses.

## Control API

The UI sends JSON commands to `/control` to manage the simulation. The current payloads are:

```json
{ "command": "pause" }
{ "command": "resume" }
{ "command": "set_speed", "value": 1.5 }
{ "command": "set_breakpoints", "tags": ["nsfw"] }
```

Each request returns the updated simulation state:

```json
{ "paused": false, "speed": 1.5, "breakpoints": ["nsfw"] }
```

## Widget Registration API

Plugins can inform the backend about available UI widgets by calling:

```http
POST /api/register_widget
{ "name": "MyWidget" }
```

The response returns the complete set of registered widget names:

```json
{ "widgets": ["MyWidget"] }
```

## Plug-in Development

External plug-ins can add new dashboard widgets without modifying the core UI. A plug-in typically serves a JavaScript bundle and registers its widget with the backend:

```http
POST /api/register_widget
{ "name": "MyWidget", "script_url": "http://localhost:5173/my_widget.js" }
```

Additional metadata keys may be included. The response contains the updated list of widget names:

```json
{ "widgets": ["MyWidget", "OtherWidget"] }
```

Once registered, the UI automatically loads the provided `script_url` so the widget behaves like a built-in panel.
On the frontend, import and call the `registerWidgetBackend` helper exported
from `culture-ui`:

```ts
import { registerWidgetBackend } from 'culture-ui/lib'

await registerWidgetBackend({
  name: 'MyWidget',
  scriptUrl: 'http://localhost:5173/my_widget.js',
})
```

### Memory Snapshots API

The backend exposes REST endpoints for listing available memory snapshot steps and retrieving snapshot data.
Use `GET /api/memory_snapshots` to list the latest steps and `GET /api/memory_snapshots/{step}` to fetch a specific snapshot.

### Agent Stats API

The Storyboard widget displays the current mood and memory retrieval count for each agent.
These values are retrieved from a new endpoint:

```http
GET /api/agent_stats
```

The response structure is:

```json
{ "agents": { "agent-1": { "mood": 0.2, "retrieval_count": 42 } } }
```


### `/stream/events` and WebSocket Fallback

The dashboard receives live updates from `/stream/events`. The preferred method uses **Server-Sent Events (SSE)**, but the UI must fall back to WebSocket when SSE isn't available.

```ts
function subscribe() {
  const sse = new EventSource('/stream/events');
  sse.onmessage = (ev) => {
    const payload = JSON.parse(ev.data);
    console.log('event', payload);
  };
  sse.onerror = () => {
    sse.close();
    const ws = new WebSocket('ws://localhost:8000/ws/events');
    ws.onmessage = (ev) => {
      const payload = JSON.parse(ev.data);
      console.log('event', payload);
    };
  };
}
```

This ensures live event delivery even when browsers or proxies block SSE.

## Setup

Install dependencies and run the development server:

```bash
pnpm install
pnpm --filter culture-ui dev
```

To execute the Playwright end-to-end tests:

```bash
pnpm --filter culture-ui build
pnpm --filter culture-ui test:e2e
```
