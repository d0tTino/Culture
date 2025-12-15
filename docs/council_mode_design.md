# Council Mode Design Overview

This document summarizes the proposed Council Mode for Culture's multi-agent simulation. It captures the goals, persona scaffolding, LangGraph orchestrator stages, success metrics, and phased milestones needed to ship the feature while reusing existing infrastructure (LangGraph orchestration, Decision Units, and Retrieval-Augmented Generation).

## Goals for Council Mode
- **Institutionalize collaborative governance:** Formalize a structured council process so agents can debate, prioritize, and ratify initiatives aligned with scenario goals instead of ad-hoc broadcasts.
- **Highlight persona diversity:** Surface contrasting viewpoints across specialized agents (Facilitator vs. Analyzer) to stress-test ideas and promote convergence.
- **Bound resource usage:** Keep DU/IP spend visible so high-agency council members remain accountable to `Ledger` controls in [`src/infra/ledger.py`](../src/infra/ledger.py).
- **Improve memory grounding:** Increase the proportion of council outputs backed by retrieved memories through nodes like [`src/agents/graphs/retriever_node.py`](../src/agents/graphs/retriever_node.py) and [`src/agents/memory/multi_layer_retriever.py`](../src/agents/memory/multi_layer_retriever.py).
- **Enable measurable progress:** Track council deliberation quality via Prometheus gauges in [`src/interfaces/metrics.py`](../src/interfaces/metrics.py) to support automated regression alerts.

## Persona Configuration Strategy
Council Mode leans on the role scaffolding that already exists in [`src/agents/core/roles.py`](../src/agents/core/roles.py):

| Persona | Primary Responsibilities | Config Levers |
| --- | --- | --- |
| **Facilitator** | Synthesize inputs, ensure inclusive debate, escalate blockers. | Prioritize `ROLE_FACILITATOR` prompt snippets and raise DU spend for deep-analysis nodes when threads stall. |
| **Innovator** | Produce divergent proposals and challenge assumptions. | Bias retrieval queries toward speculative Knowledge Board entries; assign higher weight to creative DSPy action selectors. |
| **Analyzer** | Stress-test feasibility, surface trade-offs, and call out contradictions. | Allocate DU to critique moves, request clarifications, and trigger summarization nodes when consensus drifts. |

Additional guidance:
1. **Persona packs:** Reuse `RoleProfile` generation utilities in [`roles.py`](../src/agents/core/roles.py) to seed council rosters with deterministic embeddings, making persona drift observable.
2. **Scenario binding:** Extend `DEFAULT_SCENARIO` (see [`README.md`](../README.md)) with council slots so LangGraph states know which persona is speaking during each tick.
3. **Dynamic rotation:** Build on role-swapping utilities in [`src/agents/graphs/basic_agent_graph.py`](../src/agents/graphs/basic_agent_graph.py) to rotate chair/floor roles without rebuilding prompts.

## LangGraph-Orchestrated Council Stages
Layer Council Mode atop the existing LangGraph state machine compiled in [`basic_agent_graph.py`](../src/agents/graphs/basic_agent_graph.py) and the turn controller in [`src/agents/core/base_agent.py`](../src/agents/core/base_agent.py):

1. **Docket Ingestion:** Entry nodes hydrate `AgentTurnState` with scenario directives, open proposals, and DU balances from the `Ledger`, keeping quotas current via [`src/infra/ledger.py`](../src/infra/ledger.py).
2. **Memory & Evidence Sweep:** Call `retriever_node` followed by `MultiLayerRetriever.retrieve` to assemble episodic + semantic evidence under a shared token budget, preserving token discipline while increasing RAG hit rate.
3. **Persona-Guided Deliberation:** Feed retrieved context into persona-specific reasoning policies (e.g., DSPy action selectors) so LangGraph branches (facilitation, innovation, analysis) can execute in parallel while sharing the same `AgentTurnState`.
4. **Consensus Scoring:** Aggregate persona outputs using the summarization layers in `basic_agent_graph.py` (e.g., `L1SummaryGenerator`, `L2SummaryGenerator`) so the council delivers a consolidated proposal plus dissent logs.
5. **Action & Logging:** Route approved actions to Knowledge Board updates, law proposals, or ledger transactions via `AgentController` hooks and `Ledger.log_change`, then write post-turn memory using `_should_write_post_turn_memory` policies for future retrieval.
6. **Feedback Loop:** Emit spans and metrics within each LangGraph node, reusing tracing helpers shared across memory and ledger modules.

## Fitness Metrics & Evaluation Signals
Council success should be evaluated with gauges and counters in [`src/interfaces/metrics.py`](../src/interfaces/metrics.py) plus derived composites:

- **Deliberation Throughput:** Combine `PROPOSAL_THROUGHPUT` and `ACTIVE_AGENT_COUNT` to ensure cadence stays within resource budgets.
- **RAG Fidelity:** Monitor `RAG_HIT_RATE`, `RECALL_P5`, and `P_AT_K` so council decisions cite memories retrieved in Stage 2 above a configurable threshold.
- **Resource Governance:** Track `AGENT_REMAINING_DU`, `LLM_DU_PER_1K_TOKENS`, and ledger deltas to highlight when personas exceed their authorized DU/IP envelopes.
- **Sentiment & Cohesion:** Use `AVERAGE_SENTIMENT` and `COALITION_COUNT` to detect polarization or faction clustering triggered by council debates.
- **Latency:** Watch `LLM_LATENCY_P95_MS` and `RETRIEVAL_LATENCY_P95_MS` to ensure elongated deliberations do not destabilize tick cadence.

Qualitative checks should include manual review of L1/L2 summaries and Knowledge Board diffs to confirm that council mandates are executed and dissent is captured.

## Phased Milestones
1. **Phase 0 – Stakeholder Alignment:** Circulate this design with PM/research partners, confirm KPIs, and prioritize persona coverage gaps before coding.
2. **Phase 1 – LangGraph Extensions:** Add council-specific nodes/subgraphs referencing `basic_agent_graph.py`, keeping them pluggable with the compiled graph builder and compatible with existing tracing hooks.
3. **Phase 2 – Persona & Ledger Hooks:** Wire persona packs into scenario configs, enforce DU gating for council turns via `Ledger` calls, and add UI hooks to show council rosters.
4. **Phase 3 – Memory-Rich Deliberation:** Tune retriever prompts and token budgets so council turns must pass RAG hit-rate guardrails before finalizing actions.
5. **Phase 4 – Metricized Rollout:** Build dashboards over the Prometheus metrics listed above, adding regression alerts when throughput, DU variance, or sentiment drift outside target bands.
6. **Phase 5 – Iterative Governance:** Run controlled simulations, capture Knowledge Board diffs, and iterate on persona configurations based on qualitative and quantitative feedback.

> **Stakeholder Review:** Please review this document with the designated research and product stakeholders before implementation to validate the milestones and success criteria.
