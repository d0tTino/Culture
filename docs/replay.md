# Replaying simulations

The simulation supports replaying previous runs from snapshots using the `--replay` flag. A seed can be supplied to make random behaviour deterministic.

```bash
python -m src.app --replay SNAPSHOT_PATH --seed 42
```

Using the same seed ensures Python and NumPy random number generators start from a known state.
