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

## Example Scenarios

### Simple Proposal
1. Stake a few influence points for the proposing agent (optional):
   ```bash
   curl -X POST -H 'Content-Type: application/json' \
     -d '{"agent_id": "agent_1", "amount": 5}' \
     http://localhost:8000/api/stake_ip
   ```
2. Submit a proposal:
   ```bash
   curl -X POST -H 'Content-Type: application/json' \
     -d '{"proposer_id": "agent_1", "text": "Allow concerts"}' \
     http://localhost:8000/api/governance/propose
   ```
3. Retrieve the latest proposals and vote totals:
   ```bash
   curl http://localhost:8000/api/recent_proposals
   ```

### Weighted Voting
1. Cast three "yes" votes costing nine IP:
   ```bash
   curl -X POST -H 'Content-Type: application/json' \
     -d '{"agent_id": "agent_2", "text": "Allow concerts", "approve": true, "weight": 3}' \
     http://localhost:8000/api/vote
   ```
2. Check the proposal record to confirm the additional weight:
   ```bash
   curl http://localhost:8000/api/recent_proposals
   ```

### IP Staking
An agent can lock IP to increase quadratic voting weight:
```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"agent_id": "agent_3", "amount": 10}' \
  http://localhost:8000/api/stake_ip
```
The staked amount counts toward the `sqrt(ip_balance + staked_ip)` formula used
for default voting weight.
