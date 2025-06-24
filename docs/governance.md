# Governance CLI Usage

The simulation supports submitting a law proposal before running the main event loop. Use `--proposal` and `--proposer-id` when invoking `src/app.py`.

```bash
python src/app.py --proposal "Agents must greet each other" --proposer-id agent_2
```

This calls the `forward_proposal` method on the `Simulation` instance, which delegates to `propose_law` for voting. The result is recorded on the knowledge board if approved.
