# Council Mode Design Overview

This document summarizes the proposed Council Mode for Culture's multi-agent simulation. It focuses on the objectives, persona scaffolding, LangGraph orchestrator stages, success metrics, and phased milestones needed to safely roll out the feature while reusing existing infrastructure for LangGraph flows, Decision Units (DU), and Retrieval-Augmented Generation (RAG).

## Goals for Council Mode
- **Institutionalize collaborative governance:** Formalize a structured council process so agents can debate, prioritize, and ratify initiatives aligned with scenario goals instead of relying solely on ad-hoc broadcasts.
- **Highlight persona diversity:** Surface contrasting viewpoints across specialized agents (e.g., Facilitator vs. Analyzer) to stress-test ideas and promote convergence.
- **Bound resource usage:** Keep DU/IP spend visible so high-agency council members remain accountable to `Ledger` controls in [`src/infra/ledger.py`](../src/infra/ledger.py).
- **Improve memory grounding:** Increase the proportion of council outputs backed by retrieved memories through nodes like [`src/agents/graphs/retriever_node.py`](../src/agents/graphs/retriever_node.py) and [`src/agents/memory/multi_layer_retriever.py`](../src/agents/memory/multi_layer_retriever.py).
- **Enable measurable progress:** Track council deliberation quality via existing Prometheus gauges in [`src/interfaces/metrics.py`](../src/interfaces/metrics.py), enabling automated regression alerts.

## Persona Configuration Strategy
Council Mode leans on the role scaffolding that already exists in [`src/agents/core/roles.py`](../src/agents/core/roles.py):

| Persona | Primary Responsibilities | Config Levers |
| --- | --- | --- |
| **Facilitator** | Synthesize inputs, ensure inclusive debate, escalate blockers. | Prioritize `ROLE_FACILITATOR` prompt snippets, higher DU spend for `perform_deep_analysis` nodes when threads stall. |
| **Innovator** | Produce divergent proposals and challenge assumptions. | Bias retrieval queries toward speculative knowledge board entries, assign higher weight to creative DSPy action selectors. |
| **Analyzer** | Stress-test feasibility, surface trade-offs, and call out contradictions. | Allocate DU to critique moves, request clarifications, and trigger summarization nodes when consensus drifts.

Implementation details:
1. **Persona packs:** Reuse `RoleProfile` generation utilities in [`roles.py`](../src/agents/core/roles.py) to seed council rosters with deterministic embeddings, making it easier to track persona drift.
2. **Scenario binding:** Extend the scenario definition in `DEFAULT_SCENARIO` (see [`README.md`](../README.md)) with council slots, ensuring LangGraph states know which persona is speaking.
3. **Dynamic rotation:** Build on the `AgentState` role-swapping utilities wired in [`src/agents/graphs/basic_agent_graph.py`](../src/agents/graphs/basic_agent_graph.py) to rotate chair/floor roles without rebuilding prompts.

## LangGraph-Orchestrated Council Stages
Council Mode should be layered on top of the existing state graph compiled in [`basic_agent_graph.py`](../src/agents/graphs/basic_agent_graph.py) and the turn controller in [`src/agents/core/base_agent.py`](../src/agents/core/base_agent.py):

1. **Docket Ingestion:** Use the entry nodes that hydrate `AgentTurnState` to pull scenario directives, open proposals, and DU balances from the `Ledger`. This ensures each council turn starts with up-to-date quotas from [`src/infra/ledger.py`](../src/infra/ledger.py).
2. **Memory & Evidence Sweep:** Call the async `retriever_node` followed by `MultiLayerRetriever.retrieve` to assemble episodic + semantic evidence under a shared token budget. This preserves token discipline while increasing RAG hit rate.
3. **Persona-Guided Deliberation:** Feed the retrieved context into persona-specific reasoning policies (e.g., DSPy action selectors) so LangGraph branches (facilitation, innovation, analysis) can execute in parallel while sharing the same `AgentTurnState`.
4. **Consensus Scoring:** Aggregate persona outputs using the summarization layers already wired into `basic_agent_graph.py` (e.g., `L1SummaryGenerator` and `L2SummaryGenerator`) so the council delivers a consolidated proposal plus dissent logs.
5. **Action & Logging:** Route approved actions to Knowledge Board updates, law proposals, or ledger transactions, reusing the hooks exposed on `AgentController` and `Ledger.log_change`. Each action should write post-turn memory via `_should_write_post_turn_memory` policies for future retrieval.
6. **Feedback Loop:** Emit spans and metrics within each LangGraph node, keeping compatibility with the tracing helpers already present across memory and ledger modules.

## Fitness Metrics & Evaluation Signals
Council success should be evaluated with existing gauges/counters in [`src/interfaces/metrics.py`](../src/interfaces/metrics.py) plus a few derived composites:

- **Deliberation Throughput:** Combine `PROPOSAL_THROUGHPUT` and `ACTIVE_AGENT_COUNT` to ensure the council cadence stays within resource budgets.
- **RAG Fidelity:** Monitor `RAG_HIT_RATE`, `RECALL_P5`, and `P_AT_K`—council decisions must cite memories retrieved in Stage 2 above a configurable threshold.
- **Resource Governance:** Track `AGENT_REMAINING_DU`, `LLM_DU_PER_1K_TOKENS`, and ledger deltas to highlight when personas exceed their authorized DU/IP envelopes.
- **Sentiment & Cohesion:** `AVERAGE_SENTIMENT` and `COALITION_COUNT` highlight polarization or faction clustering triggered by council debates.
- **Latency:** `LLM_LATENCY_P95_MS` and `RETRIEVAL_LATENCY_P95_MS` ensure elongated deliberations do not destabilize tick cadence.

Qualitative checks: incorporate lightweight manual review of L1/L2 summaries to judge coverage of dissent, and cross-reference Knowledge Board diff logs to ensure council mandates are executed.

## Phased Milestones
1. **Phase 0 – Stakeholder Alignment:** circulate this design with PM/research partners, confirm KPIs, and prioritize persona coverage gaps before coding.
2. **Phase 1 – LangGraph Extensions:** add council-specific nodes/subgraphs referencing `basic_agent_graph.py`, ensuring they remain pluggable with the compiled graph builder.
3. **Phase 2 – Persona & Ledger Hooks:** wire persona packs into scenario configs, enforce DU gating for council turns via `Ledger` calls, and add UI hooks to show council rosters.
4. **Phase 3 – Memory-Rich Deliberation:** tune retriever prompts and token budgets, ensuring council turns always pass RAG hit-rate guardrails before they can finalize an action.
5. **Phase 4 – Metricized Rollout:** create dashboards over the Prometheus metrics listed above, plus regression alerts when throughput, DU variance, or sentiment drift outside target bands.
6. **Phase 5 – Iterative Governance:** run controlled simulations, capture Knowledge Board diffs, and iterate on persona configurations based on qualitative + quantitative feedback.

> **Stakeholder Review:** Please review this document with the designated research and product stakeholders before implementation to validate the milestones and success criteria.
