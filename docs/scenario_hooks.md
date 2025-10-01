# Scenario Evaluation Hooks

Scenario files may register evaluation hooks that gather metrics during a
simulation run. Add the desired hooks under the `evaluation_hooks` key in the
scenario YAML and, optionally, describe the intended outcomes in
`success_metrics` and `evaluation_targets`:

```yaml
name: signature_demo
steps: 30
beats:
  - introduction
  - collaboration
  - resolution
success_metrics:
  coalition_count:
    target_max: 1
    explanation: Keep the group aligned on a single coalition.
  sentiment_curve:
    expected_trend:
      introduction: 0.05
      collaboration: 0.00
      resolution: 0.10
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
targets help flag runs that deviate from expected group dynamics. The values are
interpreted alongside the telemetry emitted by the corresponding evaluation hook.

| Hook       | Target field    | Description |
|------------|-----------------|-------------|
| `coalitions` | `max_count`     | Maximum allowed number of multi-member projects. |
| `sentiment`  | `max_variance`  | Maximum permitted variance in agent sentiment. |

Results from each hook are recorded in the metrics registry and written to the
simulation event log at the end of every beat, enabling downstream analysis or
plotting. The `success_metrics` block offers narrative guidance for scenario
authors and operators reviewing a run, while `evaluation_targets` provides
machine-readable thresholds.

### How the CLI uses success metrics and targets

Running the simulator through `python src/app.py --scenario <path>` loads the
scenario YAML, including its beats and descriptive blocks. While the CLI does not
enforce pass/fail outcomes automatically, any registered evaluation hooks write
their numeric outputs to the event log. Downstream tooling—or a quick Python
script—can compare those recorded values to the declared `success_metrics` or
`evaluation_targets` to decide whether a run stayed within the expected bounds.

### How the signature demo script surfaces compliance

`scripts/run_signature_demo.py` orchestrates `scenarios/signature_demo.yaml` end
to end. After the run finishes, the script now inspects the scenario's
`evaluation_targets`, compares them against the collected metrics, and records a
pass/fail summary in two places:

- The generated `metrics.json` contains the raw metric time series plus a
  `_target_summary` object summarizing compliance.
- The accompanying `README.md` lists the produced artifacts and adds an
  "Evaluation Target Summary" section with per-metric pass/fail notes.

## Running and replaying `signature_demo`

Run the demo and automatically export its event log and metrics:

```bash
python scripts/run_signature_demo.py
```

The script writes `event_log.jsonl`, `metrics.json`, and a human-readable
summary to `results/signature_demo/`.

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

   For an ad-hoc inspection without booting the full app CLI, use the replay
   helper instead. The tool accepts an optional `--events` flag but will also
   look for `event_log.jsonl` next to the snapshot bundle automatically:

   ```bash
   python -m tools.replay_cli demo_run/snapshots/snapshot_0.json --from 1 --to 50
   python -m tools.replay_cli demo_run/snapshots/snapshot_0.json --from 51 --to 100 \
       --events demo_run/event_log.jsonl
   ```
