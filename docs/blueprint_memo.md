# Culture.ai Blueprint Memo

This document provides a high-level summary of the major layers that make up the evolving Culture.ai architecture. These layers guide the project's long-term vision and help organize current and future work.

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

