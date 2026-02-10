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

## Discrete-Event Kernel

A **discrete-event kernel** orchestrates agent actions and environmental changes over simulated time. Rather than relying on continuous loops, events are processed in discrete steps, enabling consistent state updates and easier integration of new simulation modules.

## Memory V2

The next generation of agent memory, **Memory V2**, builds on the hierarchical summaries described in the [Hierarchical Memory System Overview](hierarchical_memory_README.md) and the more detailed [Advanced Memory Pruning Design Proposal](advanced_memory_pruning_design_proposal.md). Memory V2 aims to integrate retrieval frequency, relevance scoring, and context-aware pruning directly into the core agent workflow.

## Ledger Service

A dedicated **ledger service** records significant agent and system events for auditing and cross-agent synchronization. The ledger acts as a source of truth for inter-agent communication and can be used to replay or analyze past simulations.

## Spatial World Fabric

Culture.ai envisions a **spatial world fabric** that provides a virtual environment in which agents interact. This layer includes a grid or coordinate system, environmental rules, and mechanisms to attach memories or artifacts to specific locations. The spatial dimension enables richer world-building and more complex agent behaviors.

## Additional Components

Other major components on the roadmap include:

- **LLM Call Monitoring** for measuring performance and reliability of language model interactions.
- **Agent Personality Evolution** supported by persistent artifacts on the knowledge board.

These layers collectively define the blueprint for Culture.ai's future development.
