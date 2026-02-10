# Culture.ai Blueprint Memo

This document provides a high-level summary of the major layers that make up the evolving Culture.ai architecture. These layers guide the project's long-term vision and help organize current and future work.

## Current Capability Audit

The strategy below now reflects a repository-grounded audit of capabilities that were previously framed as mostly greenfield. Each row maps a proposed feature to concrete implementation anchors and labels the current maturity.

| Proposed feature | Repository evidence (module + class/function) | Capability label | Strategy update (replace build tasks where implemented) |
| --- | --- | --- | --- |
| Human command intake and routing | `src/interfaces/discord_bot.py` (`on_message` routes recipient/broadcast metadata), `src/sim/simulation.py` (`_handle_human_command` validates budget + dispatches) | **already implemented** | Prioritize **hardening + UX polish**: improve routing explainability for users, clearer rejection reasons (policy, rate, budget), and better operator-facing traces rather than building a new command path. |
| Discrete-event simulation control plane | `src/sim/simulation.py` (`handle_control_command`, `spawn_agent`, `retire_agent`, `_advance_world_time`) | **already implemented** | Shift to **scale + tuning**: stress-test spawn/retire churn, tune world-clock cadence defaults, and add deterministic replay checks for control-command bursts. |
| Snapshot persistence and replayability | `src/infra/snapshot.py` (`save_snapshot`, `load_snapshot`, `upload_snapshot`), `src/sim/simulation.py` (snapshot emit/load hooks) | **already implemented** | Focus on **operational hardening + observability**: integrity alerting on hash mismatches, snapshot latency metrics, storage lifecycle policies, and restore-time SLOs. |
| Knowledge-board substrate (shared memory surface) | `src/sim/knowledge_board.py` (`KnowledgeBoard`), `src/sim/graph_knowledge_board.py` (`GraphKnowledgeBoard`) | **partially implemented** | Build completion work: unify semantics between list-backed and graph-backed stores, converge query/filter behavior, and document migration/fallback rules. |
| Spatial world fabric and actions | `src/sim/world_map.py` (`WorldMap`, movement/resource effects), `src/sim/world_map_actions.py` (action wiring used by simulation turns) | **partially implemented** | Build completion work: expand map-level systems (terrain/economy/event hooks), then follow with pathfinding/perf tuning and gameplay UX polish. |
| Agent personality evolution | `src/agents/core/agent_state.py` (`PersonalityTraits`, `apply_trait_drift`) | **partially implemented** | Build completion work: connect drift signals to richer social/environmental outcomes, then add guardrails, interpretability outputs, and long-horizon balance tuning. |
| Memory V2 as a full adaptive memory program | Existing hooks touch retrieval + board state, but no complete Memory V2 pipeline is defined in the audited anchors above | **missing** | Keep as net-new build scope: define MUS/relevance lifecycle, retrieval-quality metrics, and rollout checkpoints before broad integration. |

### Immediate delivery focus from the audit

1. **Do not re-build what exists:** command routing, world clock/control plane, and snapshots should move to reliability, operator UX, and scale-validation tracks.
2. **Finish partial systems deliberately:** knowledge board parity, world-map depth, and trait-drift coupling need explicit completion milestones.
3. **Treat Memory V2 as true net-new scope:** spec, measurement, and phased rollout should be gated before broad adoption.

## Free-Form Roles

Agents in Culture.ai adopt **free-form roles**, allowing for highly flexible persona creation and dynamic roleplay interactions. This layer enables emergent behaviors and diverse agent personalities that go beyond rigid predefined job classes.

### Feature Template

1. **Existing behavior (current module/function references)**
   - `src/interfaces/discord_bot.py::on_message` parses operator text and recipient metadata before forwarding directives.
   - `src/sim/simulation.py::_handle_human_command` validates delivery constraints and dispatches role-affecting directives into the simulation.
2. **Gap/limitation**
   - Role assignment and role changes are inferred from ad hoc command wording, and rejection feedback is not yet standardized for operators.
3. **Planned change (API/schema/UX deltas only)**
   - Add a lightweight command intent schema (for example: explicit role action + target fields) at the interface boundary.
   - Return structured rejection reasons and next-step hints in Discord replies for failed role updates.

**Out of scope**
- Rewriting persona-generation internals or replacing the agent identity model.
- Replacing Discord transport or introducing a new control channel.

## Discrete-Event Kernel

A **discrete-event kernel** orchestrates agent actions and environmental changes over simulated time. Rather than relying on continuous loops, events are processed in discrete steps, enabling consistent state updates and easier integration of new simulation modules.

### Feature Template

1. **Existing behavior (current module/function references)**
   - `src/sim/simulation.py::handle_control_command`, `spawn_agent`, `retire_agent`, and `_advance_world_time` provide the current control-plane lifecycle.
2. **Gap/limitation**
   - Burst control traffic can be difficult to replay deterministically, and operator visibility into scheduling outcomes is limited.
3. **Planned change (API/schema/UX deltas only)**
   - Introduce deterministic control-command envelope metadata (sequence/time annotations) for replay and diagnostics.
   - Expand control-command acknowledgements with concise execution-state details (accepted, queued, rejected).

**Out of scope**
- Replacing the discrete-event model with a continuous simulation loop.
- Re-architecting scheduler ownership across unrelated modules.

## Memory V2

The next generation of agent memory, **Memory V2**, builds on the hierarchical summaries described in the [Hierarchical Memory System Overview](hierarchical_memory_README.md) and the more detailed [Advanced Memory Pruning Design Proposal](advanced_memory_pruning_design_proposal.md). Memory V2 aims to integrate retrieval frequency, relevance scoring, and context-aware pruning directly into the core agent workflow.

### Feature Template

1. **Existing behavior (current module/function references)**
   - Memory retrieval and board-state hooks are partially represented through simulation and shared-memory touchpoints (`src/sim/simulation.py`, `src/sim/knowledge_board.py`).
2. **Gap/limitation**
   - No end-to-end Memory V2 lifecycle contract (ingest, relevance decay, pruning, replay) is currently formalized.
3. **Planned change (API/schema/UX deltas only)**
   - Define a versioned memory-item schema with retrieval counters and relevance timestamps.
   - Publish explicit API boundaries for memory ingest/query/prune operations before implementation expansion.

**Out of scope**
- Full algorithmic implementation of Memory V2 scoring/pruning in this planning phase.
- Bulk migration of existing memory records without a staged rollout plan.

## Ledger Service

A dedicated **ledger service** records significant agent and system events for auditing and cross-agent synchronization. The ledger acts as a source of truth for inter-agent communication and can be used to replay or analyze past simulations.

### Feature Template

1. **Existing behavior (current module/function references)**
   - Snapshot and event-adjacent persistence exists in `src/infra/snapshot.py`, with simulation-level emit/load hooks in `src/sim/simulation.py`.
2. **Gap/limitation**
   - Ledger-specific event taxonomy, retention policy, and replay contracts are not yet unified under one API surface.
3. **Planned change (API/schema/UX deltas only)**
   - Define a ledger event envelope (actor, event_type, correlation_id, world_time, payload hash).
   - Specify read APIs for audit/replay queries and operator-facing filtering conventions.

**Out of scope**
- Building a production distributed log backend in this phase.
- Migrating all persistence paths to ledger-first storage immediately.

## Spatial World Fabric

Culture.ai envisions a **spatial world fabric** that provides a virtual environment in which agents interact. This layer includes a grid or coordinate system, environmental rules, and mechanisms to attach memories or artifacts to specific locations. The spatial dimension enables richer world-building and more complex agent behaviors.

### Feature Template

1. **Existing behavior (current module/function references)**
   - `src/sim/world_map.py` and `src/sim/world_map_actions.py` implement map state, movement, and resource/action effects.
2. **Gap/limitation**
   - Environmental systems and event hooks remain narrower than the target simulation depth, limiting downstream agent strategy variety.
3. **Planned change (API/schema/UX deltas only)**
   - Extend world-state schema to support additional terrain/economy/event dimensions.
   - Add action result metadata for clearer downstream simulation and interface reporting.

**Out of scope**
- A full rendering/visualization client rewrite.
- Large-scale pathfinding engine replacement during this increment.

## Additional Components

Other major components on the roadmap include:

- **LLM Call Monitoring** for measuring performance and reliability of language model interactions.
- **Agent Personality Evolution** supported by persistent artifacts on the knowledge board.

These layers collectively define the blueprint for Culture.ai's future development.

### Feature Template: LLM Call Monitoring

1. **Existing behavior (current module/function references)**
   - Monitoring signals are currently fragmented across runtime hooks rather than a dedicated monitoring contract.
2. **Gap/limitation**
   - No single schema guarantees consistent latency/error/token metrics across invocation paths.
3. **Planned change (API/schema/UX deltas only)**
   - Define a unified LLM invocation telemetry schema and dashboard-facing aggregation fields.

**Out of scope**
- Building a new external observability platform from scratch.

### Feature Template: Agent Personality Evolution

1. **Existing behavior (current module/function references)**
   - `src/agents/core/agent_state.py::apply_trait_drift` provides trait drift mechanics, with shared-memory interactions via `src/sim/knowledge_board.py` and orchestration in `src/sim/simulation.py`.
2. **Gap/limitation**
   - Trait changes are not yet consistently surfaced as explainable, user-facing deltas across simulation and interface layers.
3. **Planned change (API/schema/UX deltas only)**
   - Add structured trait-change events emitted by simulation and consumable by interface adapters.
   - Introduce a concise explanation schema for personality deltas presented to operators.

**Out of scope**
- Replacing the trait model with a wholly different psychological framework.
- Retrofitting all historical agent state snapshots in a single migration step.
