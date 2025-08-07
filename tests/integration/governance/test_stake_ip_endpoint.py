import importlib
import json
from pathlib import Path

import pytest

from src.infra.ledger import Ledger
from src.interfaces import dashboard_backend as db


@pytest.mark.asyncio
@pytest.mark.integration
async def test_stake_ip_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite")
    gservice = importlib.import_module("src.governance.service")
    monkeypatch.setattr(gservice, "ledger", ledger)
    monkeypatch.setattr(db, "governance", gservice.governance)

    resp = await db.api_stake_ip(db.StakeRequest(agent_id="a1", amount=5.0))
    data = json.loads(resp.body)
    assert data["staked_ip"] == pytest.approx(5.0)
