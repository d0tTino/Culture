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

During the run, evaluation events are emitted at the end of each beat, recording
coalition counts and sentiment snapshots for plotting.

## Scripted Beats

1. **Proposal** – the planner presents a project idea.
2. **Critique** – the critic challenges and refines the plan.
3. **Vote** – participants decide whether to proceed.
4. **Deliverable** – the worker produces the agreed output.

## Metrics

- **Coalitions formed**: number of projects with more than one member.
- **Sentiment curves**: average agent mood over time.

The resulting plots appear in the `plots/` directory.

## Signature Demo Scenario

This lightweight scenario highlights evaluation hooks for common group metrics.

### Usage

1. Run the simulation with the signature demo:
   ```bash
   python src/app.py --scenario scenarios/signature_demo.yaml
   ```
2. Export trace data:
   ```bash
   python scripts/export_traces.py --snapshots snapshots --output data/traces.jsonl
   ```
3. Generate plots and bundle metrics:
   ```bash
   python tools/export_traces.py data/traces.jsonl --outdir plots --bundle run_bundle
   ```

### Expected Arc

1. **Introduction** – agents share initial perspectives and set intent.
2. **Collaboration** – members coordinate to sign onto a shared plan.
3. **Resolution** – the group finalizes the plan and reflects on the process.

Evaluation hooks record coalition counts, sentiment, and collective DU/IP curves for analysis.
