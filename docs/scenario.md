# Scenario Guide

Culture ships with curated simulation blueprints to cover the most common collaboration patterns. Each scenario below references a YAML file in the `scenarios/` directory that you can pass to `src/app.py`.

```bash
python src/app.py --scenario <path-to-scenario>
```

## Demo Warm-up (`scenarios/demo.yaml`)
- **Use when:** You need a smoke test to confirm agents start, exchange messages, and record snapshots.
- **Pacing:** Two agents, five steps.
- **Why it exists:** Validates new environments or CI images without the overhead of additional hooks.

## Planning Project (`scenarios/planning_project.yaml`)
- **Use when:** You want a longer-form planning exercise with distinct proposal → critique → vote → deliverable beats.
- **Key beats:**
  1. **Proposal** – the planner presents a project idea.
  2. **Critique** – the critic challenges and refines the plan.
  3. **Vote** – participants decide whether to proceed.
  4. **Deliverable** – the worker produces the agreed output.
- **Instrumentation:** Simple coalition and sentiment success metrics for quick retros.

Run it with:
```bash
python src/app.py --scenario scenarios/planning_project.yaml
```

## Signature Demo (`scenarios/signature_demo.yaml`)
- **Use when:** You need end-to-end evaluation artifacts, including coalition tracking, sentiment variance, and collective DU/IP deltas.
- **Beats:** Proposal, critique, vote, deliverable with narrative guidance embedded in the YAML file.
- **Companion script:** `python scripts/run_signature_demo.py` orchestrates exports, bundle generation, and README updates.

## Crisis Response (`scenarios/crisis_response.yaml`)
- **Use when:** Stress-testing communication during incident drills across alert, triage, stabilization, and recovery phases.
- **Highlights:**
  - Narrative beats encourage calm coordination while sentiment steadily recovers.
  - Success metrics watch for splinter coalitions, sentiment variance, and response alignment milestones.
- **Expected outcome:** A replay slice that documents the mitigation timeline plus metrics to audit decision velocity.

## Research Sprint (`scenarios/research_sprint.yaml`)
- **Use when:** Facilitating focused discovery work where agents explore, experiment, synthesize, and publish findings in a tight loop.
- **Highlights:**
  - Encourages divergent idea generation before converging on a publishable insight.
  - Tracks coalition stability, positive sentiment trends, and minimum knowledge updates.
- **Expected outcome:** Bundled traces capture experiment logs and the final report outline for follow-up analysis.
