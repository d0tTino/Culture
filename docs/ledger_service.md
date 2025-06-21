# Ledger Service

The ledger provides a persistent record of IP and DU adjustments for each agent.
It uses a small SQLite database in WAL mode located at `ledger.sqlite3` by
default.

## Usage

```python
from src.infra.ledger import ledger

# record changes
ledger.log_change("agent-1", delta_ip=1.5, delta_du=-2, reason="message")

# query balances
ip, du = ledger.get_balance("agent-1")
```

Transactions are stored in the `transactions` table while current balances are
kept in `agent_balances`. The ledger is lightweight and requires no setup other
than importing the module.
