# Council Mode Design Notes

Council Mode is an opt-in orchestration pattern where multiple specialized agents debate or critique a proposal before the simulation executes it. The goal is to increase robustness for high-stakes actions while keeping the feature experimental and easy to disable.

## Current Status
- **Experimental:** Disabled by default. Enable only when iterating on council prompts and evaluation workflows.
- **Pluggable:** Implemented as a higher-level protocol that wraps existing LangGraph flows.
- **Guardrailed:** Each councilor must operate within strict budgets to avoid runaway token usage.

## Next Steps
1. Add scenario templates that showcase council deliberation loops.
2. Capture telemetry for the entire council session to simplify debugging.
3. Harden failure recovery so the main simulation can proceed if the council deadlocks.
