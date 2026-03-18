# Culture UI

This package houses the React + TypeScript front-end for the Culture project. It is managed from the monorepo root using pnpm workspaces.

## Install dependencies

Run pnpm from the repository root to install all workspace packages:

```bash
pnpm install
```

## Development server

Start the UI in development mode with:

```bash
pnpm --filter culture-ui dev
```

This launches Vite at `http://localhost:5173` by default.

## Build

Create an optimized production build:

```bash
pnpm --filter culture-ui build
```

Output files are written to `culture-ui/dist`.

## Lint

Check the UI source with ESLint:

```bash
pnpm --filter culture-ui lint
```

## Type check

Verify the codebase with the TypeScript compiler:

```bash
pnpm --filter culture-ui type-check
```

## Format

Automatically format source files with Prettier:

```bash
pnpm --filter culture-ui format
```

## Git hooks

Husky runs `pnpm lint` and `pnpm type-check` before each commit.

## End-to-end tests

Run Playwright tests against the built UI:

```bash
pnpm --filter culture-ui build
pnpm --filter culture-ui test:e2e
```

Run the entire Playwright suite or target a single file:

```bash
# all tests
pnpm --filter culture-ui test:e2e

# specific test
pnpm --filter culture-ui test:e2e tests/memory-explorer.pw.ts
```

Use `pnpm` to execute tests so that the bundled Playwright version matches the
installed `@playwright/test` dependency. Running `npx playwright` may install a
different version and cause failures.

## Workspace integration

`culture-ui` is defined in `pnpm-workspace.yaml`. Running `pnpm install` at the root installs both backend and UI dependencies. Use `--filter culture-ui` to run scripts only for the UI when needed.

## Mission Overview & Agent Data Overview

The UI includes pages for monitoring active missions and reviewing agent data:

- **Mission Overview** – shows current missions with status and progress for each agent.
- **Agent Data Overview** – lists observations, messages and other data gathered by agents.
- **Proposals** – view recent law proposals and vote results (navigate to `/proposals`).

Screenshots will be added to this README as these pages mature.

## Storyboard Widget

The Storyboard widget streams simulation events from `/ws/events` and
displays agent coordinates and current mood values. A "Summaries" tab fetches
recent memory summaries for the selected agent via
`/api/agents/{id}/semantic_summaries`.

Memory Explorer features an agent selector input that reloads summaries whenever
the chosen ID changes.

Run the development server and navigate to `/storyboard` to see it in action.

## LiveMap Widget

`LiveMap` listens to `/api/map/stream` via Server-Sent Events and renders the
latest agent positions. The widget also displays each agent's mood and a recent
one-line summary. Register it in your dashboard layout using the widget name
`LiveMap`.


## User Value KPI reference

The KPI Card page (`/kpi-card`) includes the canonical field definitions for the
`/api/user_value_metrics` payload. Treat the repository root README
(`../README.md#user-value-kpi-reference`) as the engineering source of truth and
this page as the in-dashboard companion reference for product reviews.

