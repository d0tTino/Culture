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

## Workspace integration

`culture-ui` is defined in `pnpm-workspace.yaml`. Running `pnpm install` at the root installs both backend and UI dependencies. Use `--filter culture-ui` to run scripts only for the UI when needed.

## Mission Overview & Agent Data Overview

The UI includes pages for monitoring active missions and reviewing agent data:

- **Mission Overview** – shows current missions with status and progress for each agent.
- **Agent Data Overview** – lists observations, messages and other data gathered by agents.

Screenshots will be added to this README as these pages mature.
