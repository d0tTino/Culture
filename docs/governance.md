# Governance CLI Usage

The simulation supports submitting a law proposal before running the main event loop. Use `--proposal` and `--proposer-id` when invoking `src/app.py`.

```bash
python src/app.py --proposal "Agents must greet each other" --proposer-id agent_2
```

This calls the `forward_proposal` method on the `Simulation` instance, which delegates to `propose_law` for voting. The result is recorded on the knowledge board if approved.

Votes are weighted by the amount of influence points (IP) agents have staked. Past proposals and their outcomes can be viewed using the dashboard API:

```bash
python examples/governance/list_proposals_example.py
```

The core voting logic lives in ``src.governance.service`` where
``GovernanceService`` exposes methods for proposing laws, weighting votes via
the ledger, and retrieving previous proposals.

The default weight for an agent is computed with a quadratic formula:

```
weight = sqrt(ip_balance + staked_ip)
```

For instance, an agent with ``9`` IP and ``7`` IP staked contributes
``sqrt(16) = 4`` votes.

## Weighted Votes

Additional votes can be submitted by passing `--vote-weights` with a comma
separated list of `agent_id=weight` pairs. Each extra vote costs its square in
IP, following the formula ``cost = votes^2``. For example, casting three votes
spends ``3^2 = 9`` IP. To give `agent_1` three votes and `agent_2` a single vote:

```bash
python src/app.py \
  --proposal "Allow concerts" --proposer-id agent_1 \
  --vote-weights agent_1=3,agent_2=1
```

## Dashboard API

The dashboard backend exposes a few read-only endpoints for governance data.

### `GET /api/laws`
Returns a list of laws that have been passed and recorded on the law board.

### `GET /api/votes`
Lists recent law proposals with their vote totals as stored in the ledger.
