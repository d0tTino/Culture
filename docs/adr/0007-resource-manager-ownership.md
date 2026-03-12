# ADR-0007: Canonical Resource Manager Ownership Boundaries

- Status: Accepted
- Date: 2026-03-11

## Context

Resource accounting responsibilities were split across `src/sim/resource_manager.py` and
`src/agents/core/resource_manager.py`. That duplication blurred ownership of DU/IP caps,
budget validation/charging, and ledger side effects.

This ambiguity caused three recurring risks:

1. Agent and simulation pipelines could evolve different accounting behaviors.
2. LLM and action paths used implicit methods (`ensure_du_budget`, `charge_du`) without an
   explicit contract.
3. Concurrent command/application paths risked violating DU invariants if charge logic was not
   centralized and atomic.

## Decision

1. `src/sim/resource_manager.py` is the **sole canonical implementation** for:
   - per-tick IP/DU cap enforcement,
   - DU budget checks,
   - DU charging,
   - resource-side ledger and event side effects.
2. `src/agents/core/resource_manager.py` becomes a **compatibility re-export** that forwards
   to canonical `sim` symbols.
3. Canonical explicit interfaces are introduced and consumed by call sites:
   - `BudgetChecker` via `budget_check(...)`,
   - `BudgetCharger` via `charge(...)`,
   - `TickCapper` via `tick_cap(...)`.
4. DU mutation operations in the canonical module are protected with a lock to preserve atomic
   invariants under concurrent callers.

## Ownership boundaries

- `sim`: owns resource policy and state transitions (caps, budget checks, charges).
- `agents`: consumes resource interfaces; does not define competing resource logic.
- `infra/ledger`: remains the financial/event sink for durable recording; it does not own DU/IP
  policy decisions.

## Consequences

- Positive:
  - One source of truth for DU/IP accounting.
  - Explicit contracts simplify call-site intent and testing.
  - Concurrent paths maintain non-negative budget invariants.
- Tradeoffs:
  - Legacy APIs are retained as aliases for compatibility and should be phased out gradually.
  - Coupling to canonical module increases migration pressure on any lingering duplicate imports.
