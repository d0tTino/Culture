# Council Mode Design Overview

This document summarizes the proposed Council Mode for Culture's multi-agent simulation. It captures the goals, persona scaffolding, LangGraph orchestrator stages, success metrics, and phased milestones needed to ship the feature while reusing existing infrastructure (LangGraph orchestration, Decision Units, and Retrieval-Augmented Generation).

## Problem Statement
Implement local council mode (multiple AI agents + judge) on a single GPU.

## Goals for Council Mode
- **Institutionalize collaborative governance:** Formalize a structured council process so agents can debate, prioritize, and ratify initiatives aligned with scenario goals instead of ad-hoc broadcasts.
- **Highlight persona diversity:** Surface contrasting viewpoints across specialized agents (Facilitator vs. Analyzer) to stress-test ideas and promote convergence.
- **Bound resource usage:** Keep DU/IP spend visible so high-agency council members remain accountable to `Ledger` controls in [`src/infra/ledger.py`](../src/infra/ledger.py).
- **Improve memory grounding:** Increase the proportion of council outputs backed by retrieved memories through nodes like [`src/agents/graphs/retriever_node.py`](../src/agents/graphs/retriever_node.py) and [`src/agents/memory/multi_layer_retriever.py`](../src/agents/memory/multi_layer_retriever.py).
- **Enable measurable progress:** Track council deliberation quality via Prometheus gauges in [`src/interfaces/metrics.py`](../src/interfaces/metrics.py) to support automated regression alerts.

### Single-GPU performance & DU targets
- **Latency (A100-class, 3 members, 1 round):** p95 end-to-end council latency ≤ **1.5s** with member fan-out ≤ **0.75s**.
- **Throughput:** Sustain **≥25 questions/minute** on a single GPU with the default 3-member roster (judge + members).
- **DU ceilings:** Cap per-member DU at **2.0** (≤**6 DU** aggregate for the default roster) and reserve **≤1 DU** for the judge step; runs that would exceed these ceilings should short-circuit with partial outcomes.

## Component Overview (Spec Alignment)
Council Mode is composed of the following spec-aligned components and responsibilities:

- **CouncilMemberConfig:** Per-member configuration (name, role, model, temperature, DU/IP budgets, and any tool restrictions) used to instantiate each council agent deterministically.
- **CouncilConfig:** Top-level configuration describing roster, concurrency limits, shared DU budgets, and default settings for council runs.
- **CouncilQuestion:** Structured input payload that packages the question, context, RAG documents, and a stable `question_id` for metrics/logging.
- **CouncilOutcome:** Structured output capturing the winning answer, dissenting notes, judge rationale, and references to evidence used.
- **CouncilOrchestrator:** The runtime coordinator that spawns members, dispatches turns, collects responses, and invokes the judge.
- **CLI:** Entry-point for running a council question locally, wiring `CouncilConfig` + `CouncilQuestion` and emitting the `CouncilOutcome`.
- **LangGraph node:** Dedicated node/subgraph that wraps council execution so it can be invoked from existing agent flows.
- **Metrics:** Prometheus counters/gauges for question throughput, DU spend, latency, and judge outcome labeling.

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
  USE_COUNCIL_MODE=true make council Q="Should we prioritize the supply-chain audit?"
  ```
- **Direct CLI for richer context:**
  ```bash
  USE_COUNCIL_MODE=true \
  python -m scripts.council_cli \
    "Should we prioritize the supply-chain audit?" \
    --context "Procurement stalled last sprint" \
    --rag-doc "Incident INC-2045" \
    --rag-doc "Q3 vendor scorecard" \
    --question-id "audit-priority-check"
  ```
  The CLI accepts repeated `--rag-doc/--rag-docs` flags to supply additional evidence, and `--question-id` is persisted in metrics to link answers to observability traces.

> If `USE_COUNCIL_MODE` is omitted or set to `false`, the CLI exits with a guardrail message and status 1 by default; use `--bypass-env-guard` to override.

### Council-specific toggles
- `USE_COUNCIL_MODE`: Enable/disable council orchestration globally; set in `.env` or via `export USE_COUNCIL_MODE=true` before running simulations.
- `COUNCIL_CONFIG_PATH`: YAML roster location (default `config/council.yml`). Use to point at environment-specific persona rosters without code changes.
- `COUNCIL_MAX_CONCURRENT_CALLS`: Maximum concurrent LLM invocations per council run, used to bound latency and spend.
- `DU_BUDGET_PER_QUESTION`: DU envelope shared by all council members for a single prompt.
- `ROLE_DU_GENERATION`: Per-persona DU generation settings that control how the Facilitator/Innovator/Analyzer accumulate budget across ticks.
- `config/council.yml`: Example roster showing `max_concurrent_calls`, `du_budget_per_question`, and `members`. Member `model` values are optional and default to `DEFAULT_LLM_MODEL` when omitted; override values here or through environment variables consumed in [`src/infra/config.py`](../src/infra/config.py) and [`src/infra/settings.py`](../src/infra/settings.py).

When the YAML file is missing or malformed, Culture falls back to defaults emitted by `_build_default_council_config`, so corrupted configs do not block simulations.

### PewDiePie-style roster (YouTube-friendly experiment)
A creator-inspired loadout can help stress-test banter, cross-talk, and audience-facing recaps. Save the snippet below as `config/pewdiepie_council.yml` and point `COUNCIL_CONFIG_PATH` at it to try the roster:

```yaml
max_concurrent_calls: 3
du_budget_per_question: 6.0
voting_mode: judge_llm
members:
  - member_id: bro-facilitator
    display_name: Bro Facilitator
    role: Facilitator
    persona: Keeps the pacing high, summarizes takes with signature "Bro Army" hype, and calls on others quickly.
    temperature: 0.35
    max_tokens: 256
    is_active: true
  - member_id: meme-engineer
    display_name: Meme Engineer
    role: Innovator
    persona: Drops punchy meme riffs and wildcard pivots to keep ideation lively, while citing receipts.
    temperature: 0.55
    max_tokens: 256
    is_active: true
  - member_id: zero-deaths-critic
    display_name: Zero Deaths Critic
    role: Analyzer
    persona: Applies "zero deaths" rigor to poke holes, spot contradictions, and demand clear receipts.
    temperature: 0.25
    max_tokens: 256
    is_active: true
```

Sample CLI invocation (enables council mode, swaps to the PewDiePie roster, and pins a question ID for observability):

```bash
USE_COUNCIL_MODE=true \
COUNCIL_CONFIG_PATH=config/pewdiepie_council.yml \
python -m scripts.council_cli "Is the Zero Deaths meme still on-brand?" --question-id "pewdiepie-zero-deaths"
```

To revert, unset or remove `COUNCIL_CONFIG_PATH` so it falls back to `config/council.yml`, or set `USE_COUNCIL_MODE=false` to return to the standard single-persona run.

## Phased Milestones
1. **Phase 1 – Council config scaffolding:** Define `CouncilMemberConfig`, `CouncilConfig`, and `CouncilQuestion` data structures, plus YAML loading/validation.  
   **Readiness:** Configuration round-trips from YAML to runtime objects with validation errors surfaced clearly.
2. **Phase 2 – Local orchestrator + judge:** Implement `CouncilOrchestrator` with local multi-agent execution and a judge step that outputs `CouncilOutcome`.  
   **Readiness:** Local runs complete end-to-end on a single GPU with deterministic member ordering and a captured judge rationale.
3. **Phase 3 – CLI wiring:** Deliver the council CLI to build `CouncilQuestion` inputs, execute the orchestrator, and print/serialize outcomes.  
   **Readiness:** CLI supports context, RAG docs, and question IDs; outputs include verdict, dissent, and evidence references.
4. **Phase 4 – LangGraph node integration:** Wrap council execution in a dedicated LangGraph node or subgraph to plug into existing flows.  
   **Readiness:** Council node composes with `basic_agent_graph.py` without breaking existing tracing or turn state.
5. **Phase 5 – Metrics & observability:** Instrument metrics for throughput, latency, DU spend, and judge outcomes, keyed by `question_id`.  
   **Readiness:** Dashboards surface council runs with per-member labels and stable `question_id` filters.
6. **Phase 6 – Memory & RAG alignment:** Ensure council deliberation uses the retriever stack and captures evidence in outcomes.  
   **Readiness:** RAG hit-rate guardrails pass and outcomes cite retrieved sources.
7. **Phase 7 – Resource governance:** Enforce DU/IP budgets and concurrency limits in council runs, aligned with ledger controls.  
   **Readiness:** Budget overruns are blocked or logged, and ledger deltas reconcile with council activity.
8. **Phase 8 – Production readiness:** Validate configuration hygiene, operational runbooks, and failure containment.  
   **Readiness:** Fallback configs work, CLI runbooks are documented, and end-to-end runs are stable under fault injection.

> **Stakeholder Review:** Please review this document with the designated research and product stakeholders before implementation to validate the milestones and success criteria.
