# Deterministic replay

The simulation can be replayed deterministically by combining snapshots, event logs and a fixed random seed.

## Running with a stable seed

Provide `--seed` when running a simulation to initialise Python and NumPy random number generators. The seed is recorded alongside each logged event and the RNG state is stored in every snapshot.

```bash
python -m src.app --seed 42
```

## Replaying from a snapshot

To replay a previous run, pass the snapshot path to `--replay`. Optional `--replay-start` and `--replay-end` flags restrict which ticks from the event log are applied. Supplying the same `--seed` as the original run restores deterministic behaviour.

```bash
python -m src.app --replay snapshots/snapshot_10.json --seed 42 \
    --replay-start 11 --replay-end 20
```

## Slicing event logs

Event logs can be sliced by tick range using `tools/export_traces.py`:

```bash
python tools/export_traces.py traces.jsonl --start 11 --end 20
```

This produces plots and optional replay bundles containing only the selected range of events.

## Logging misbehavior

Misbehavior events can be recorded separately for audit purposes:

```python
from src.infra import event_log

event_log.log_misbehavior({"step": 42, "detail": "unexpected action"})
```

These entries include the simulation seed, previous event hash and a trace hash.
They can be retrieved via `fetch_events`:

```python
mis = event_log.fetch_events(event_type="misbehavior")
```

