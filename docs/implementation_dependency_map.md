# Implementation Dependency Map and ADR Checkpoints

This document defines the required implementation order for high-impact platform work
so multiple teams can ship in parallel without creating incompatible behavior.

## Prerequisite Layers

The architecture must be implemented in this order:

1. **Interaction bus**
2. **State consistency**
3. **Memory/query semantics**
4. **Governance rules**
5. **Scale-out**

Downstream layers may not be merged until upstream layer checkpoints are accepted.

## Phase-by-Phase ADR Checkpoints (required before coding)

For each phase below, open and approve an ADR before any high-impact code changes begin.

### 1) Interaction bus

**Goal:** Define event envelope, step dispatch boundaries, and ordering guarantees.

**ADR checkpoints:**
- Event envelope schema (required fields, optional extensions, versioning strategy).
- Delivery semantics (at-most-once / at-least-once expectations for each event type).
- Event lifecycle boundaries for producer/consumer ownership.
- Backward-compatibility and migration strategy for event names/payloads.

### 2) State consistency

**Goal:** Ensure all replicated state transitions are deterministic and replayable.

**ADR checkpoints:**
- Canonical state transition rules and idempotency expectations.
- Conflict handling strategy (merge policy, last-write-wins exceptions, vector clocks).
- Snapshot/replay compatibility requirements and minimum validation checks.
- Rollback/failure handling rules for partially applied changes.

### 3) Memory/query semantics

**Goal:** Lock read/write semantics so memory behavior is stable across implementations.

**ADR checkpoints:**
- Query contract (inputs, ranking expectations, token caps, fallback behavior).
- Write contract (normalization, deduplication, summarization, retention rules).
- Schema compatibility rules for memory payload evolution.
- Observability requirements for memory correctness (metrics + traces).

### 4) Governance rules

**Goal:** Apply policy controls on top of stable interaction/state/memory layers.

**ADR checkpoints:**
- Policy evaluation boundaries (what is governed, where decisions are enforced).
- Conflict resolution between policy outcomes and simulation progression.
- Auditability requirements (decision logs, policy versions, replayability).
- Exception/override model and who can authorize it.

### 5) Scale-out

**Goal:** Scale throughput and topology without changing layer contracts.

**ADR checkpoints:**
- Partitioning/sharding strategy and routing rules.
- Cross-partition consistency model and acceptable staleness.
- Load-shedding/backpressure controls and SLO-based safeguards.
- Upgrade/rollout strategy that preserves compatibility with existing contracts.

## Change Control Rule

Before implementing high-impact changes, teams must verify:

- The current phase ADR exists and is approved.
- All prerequisite phase ADRs are approved.
- The change does not violate any must-not-change contracts in:
  - `src/sim/simulation.py`
  - `src/sim/knowledge_board.py`
  - `src/agents/core/agent_state.py`

