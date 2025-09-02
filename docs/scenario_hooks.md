# Scenario Evaluation Hooks

Scenario files may register evaluation hooks that gather metrics during a
simulation run. Add the desired hooks under the `evaluation_hooks` key in the
scenario YAML:

```yaml
name: signature_demo
steps: 30
beats:
  - introduction
  - collaboration
  - resolution
evaluation_hooks:
  - coalitions
  - sentiment
  - collective_du
  - collective_ip
evaluation_targets:
  coalitions:
    max_count: 0
  sentiment:
    max_variance: 0.1
```

The following built-in hooks are available:

| Hook name       | Description |
|-----------------|-------------|
| `coalitions`    | Number of projects with more than one member. |
| `sentiment`     | Average agent mood level. |
| `collective_du` | Total DU (durability units) across agents. |
| `collective_ip` | Total IP (influence points) across agents. |

## Evaluation targets

Use the optional `evaluation_targets` block to declare bounds for each metric. These
targets help flag runs that deviate from expected group dynamics.

| Hook       | Target field    | Description |
|------------|-----------------|-------------|
| `coalitions` | `max_count`     | Maximum allowed number of multi-member projects. |
| `sentiment`  | `max_variance`  | Maximum permitted variance in agent sentiment. |

Results from each hook are recorded in the metrics registry and written to the
simulation event log at the end of every beat, enabling downstream analysis or
plotting.

## Running and replaying `signature_demo`

1. Run the scenario and generate snapshots and an event log (default
   `event_log.jsonl`):

   ```bash
   python src/app.py --scenario scenarios/signature_demo.yaml --seed 42
   ```

2. Package the run’s artifacts for sharing or replay:

   ```bash
   python scripts/export_traces.py --events event_log.jsonl \
       --snapshots-dir snapshots -o traces.jsonl --bundle signature_demo.zip
   ```

   The resulting `signature_demo.zip` contains the event log, snapshots,
   metrics, and exported traces.

3. Replay the bundled run:

   ```bash
   unzip signature_demo.zip -d demo_run
   python src/app.py --replay demo_run/snapshots/snapshot_0.json --seed 42
   ```

   Adjust `--replay-start` and `--replay-end` to slice the run if desired.
