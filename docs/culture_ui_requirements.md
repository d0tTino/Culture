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
