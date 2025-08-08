# Planning Project Scenario

This scenario guides a small group through a structured planning process.

## Setup

1. Run the simulation using the scenario file:
   ```bash
   python src/app.py --scenario scenarios/planning_project.yaml
   ```
2. Export trace data after the run:
   ```bash
   python scripts/export_traces.py --snapshots snapshots --output data/traces.jsonl
   ```
3. Generate evaluation plots:
   ```bash
   python tools/export_traces.py data/traces.jsonl --outdir plots
   ```

## Scripted Beats

1. **Proposal** – the planner presents a project idea.
2. **Critique** – the critic challenges and refines the plan.
3. **Vote** – participants decide whether to proceed.
4. **Deliverable** – the worker produces the agreed output.

## Metrics

- **Coalitions formed**: number of projects with more than one member.
- **Sentiment curves**: average agent mood over time.

The resulting plots appear in the `plots/` directory.
