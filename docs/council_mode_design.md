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

### Fitness Inspection Guidance
- **On-demand inspection via CLI:** Run the council CLI (see examples below) with `--question-id` to correlate winning personas, DU spend, and resolution text with Prometheus timeseries during triage.
- **Metrics lens:** Filter `PROPOSAL_THROUGHPUT`, `RAG_HIT_RATE`, `AGENT_REMAINING_DU`, and `LLM_LATENCY_P95_MS` by `question_id` and `member_id` labels to see which personas drive most DU/IP consumption and whether retrieval discipline holds under load.
- **Ledger & Knowledge Board diffs:** Cross-check DU/IP balances in the `Ledger` logs against Knowledge Board entries created after council resolutions to validate that governance decisions execute and are retained for later retrieval.
- **Drift detection:** Compare sentiment/coalition gauges before and after a council cycle to detect polarization regressions, especially when persona packs or prompt scaffolds change.

## CLI Usage & Configuration Flags

### Quickstarts
- **Makefile helper (no config changes):**
  ```bash
  make council Q="Should we prioritize the supply-chain audit?"
  ```
- **Direct CLI for richer context:**
  ```bash
  python -m scripts.council_cli \
    "Should we prioritize the supply-chain audit?" \
    --context "Procurement stalled last sprint" \
    --rag-doc "Incident INC-2045" \
    --rag-doc "Q3 vendor scorecard" \
    --question-id "audit-priority-check"
  ```
  The CLI accepts repeated `--rag-doc/--rag-docs` flags to supply additional evidence, and `--question-id` is persisted in metrics to link answers to observability traces.

### Council-specific toggles
- `USE_COUNCIL_MODE`: Enable/disable council orchestration globally; set in `.env` or via `export USE_COUNCIL_MODE=true` before running simulations.
- `COUNCIL_CONFIG_PATH`: YAML roster location (default `config/council.yml`). Use to point at environment-specific persona rosters without code changes.
- `COUNCIL_MAX_CONCURRENT_CALLS`: Maximum concurrent LLM invocations per council run, used to bound latency and spend.
- `DU_BUDGET_PER_QUESTION`: DU envelope shared by all council members for a single prompt.
- `ROLE_DU_GENERATION`: Per-persona DU generation settings that control how the Facilitator/Innovator/Analyzer accumulate budget across ticks.
- `config/council.yml`: Example roster showing `max_concurrent_calls`, `du_budget_per_question`, and `members` with per-role models; override values here or through environment variables consumed in [`src/infra/config.py`](../src/infra/config.py) and [`src/infra/settings.py`](../src/infra/settings.py).

When the YAML file is missing or malformed, Culture falls back to defaults emitted by `_build_default_council_config`, so corrupted configs do not block simulations.

## Phased Milestones
1. **Phase 0 – Stakeholder Alignment:** Circulate this design with PM/research partners, confirm KPIs, and prioritize persona coverage gaps before coding.
2. **Phase 1 – LangGraph Extensions:** Add council-specific nodes/subgraphs referencing `basic_agent_graph.py`, keeping them pluggable with the compiled graph builder and compatible with existing tracing hooks.
3. **Phase 2 – Persona & Ledger Hooks:** Wire persona packs into scenario configs, enforce DU gating for council turns via `Ledger` calls, and add UI hooks to show council rosters.
4. **Phase 3 – Memory-Rich Deliberation:** Tune retriever prompts and token budgets so council turns must pass RAG hit-rate guardrails before finalizing actions.
5. **Phase 4 – Metricized Rollout:** Build dashboards over the Prometheus metrics listed above, adding regression alerts when throughput, DU variance, or sentiment drift outside target bands.
6. **Phase 5 – Iterative Governance:** Run controlled simulations, capture Knowledge Board diffs, and iterate on persona configurations based on qualitative and quantitative feedback.

### Phase 8 – Readiness Checklist
Track these blocking items before declaring Council Mode production-ready:

- [ ] **Configuration hygiene:** `USE_COUNCIL_MODE` gating verified in staging, `COUNCIL_CONFIG_PATH` points to a validated roster, and DU envelopes (`DU_BUDGET_PER_QUESTION`, `COUNCIL_MAX_CONCURRENT_CALLS`) are tuned for your LLM capacity.
- [ ] **Observability baseline:** Prometheus dashboards chart `PROPOSAL_THROUGHPUT`, `RAG_HIT_RATE`, `AGENT_REMAINING_DU`, sentiment/coalition gauges, and P95 latency with `question_id` filters wired through the CLI.
- [ ] **Retrieval discipline:** Sampled council runs meet minimum RAG hit rate and cite retrieved evidence in L1/L2 summaries; Knowledge Board diffs capture resolutions plus dissent.
- [ ] **Failure containment:** Fallback persona roster loaded from `_build_default_council_config` confirmed in chaos tests (missing YAML, timeouts), and ledger audits show DU/IP debits for cancelled or timed-out calls.
- [ ] **UX/operational playbooks:** CLI runbook documents context, RAG doc, and question ID usage; stakeholders trained to trace a resolution from CLI input to Knowledge Board entry and ledger deltas.

> **Stakeholder Review:** Please review this document with the designated research and product stakeholders before implementation to validate the milestones and success criteria.
