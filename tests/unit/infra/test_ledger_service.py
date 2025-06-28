import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def load_service():
    if "fastapi" in sys.modules:
        fastapi_mod = sys.modules["fastapi"]
    else:
        fastapi_mod = types.ModuleType("fastapi")
        sys.modules["fastapi"] = fastapi_mod
    if not hasattr(fastapi_mod, "FastAPI") or not hasattr(getattr(fastapi_mod, "FastAPI"), "post"):

        class _FastAPI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

            def post(self, *args: object, **kwargs: object):
                def dec(fn):
                    return fn

                return dec

        fastapi_mod.FastAPI = _FastAPI
        responses_mod = types.ModuleType("fastapi.responses")

        class _JSONResponse:
            def __init__(self, content, *a, **k) -> None:
                self.body = json.dumps(content).encode("utf-8")

        responses_mod.JSONResponse = _JSONResponse
        sys.modules["fastapi.responses"] = responses_mod
    if "uvicorn" not in sys.modules:
        uvicorn_mod = types.ModuleType("uvicorn")
        uvicorn_mod.run = lambda *a, **k: None
        sys.modules["uvicorn"] = uvicorn_mod
    if "src.infra.ledger_service" in sys.modules:
        del sys.modules["src.infra.ledger_service"]
    return importlib.import_module("src.infra.ledger_service")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reward_and_spend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = load_service()
    from src.infra.ledger import Ledger

    test_ledger = Ledger(tmp_path / "ledger.sqlite")
    monkeypatch.setattr(svc, "ledger", test_ledger)

    req = svc.RewardRequest(agent_id="a", ip=2.0, du=3.0)
    resp = await svc.reward(req)
    data = json.loads(resp.body)
    assert data["ip"] == pytest.approx(2.0)
    assert data["du"] == pytest.approx(3.0)

    spend_req = svc.SpendRequest(agent_id="a", du=1.0)
    resp = await svc.spend(spend_req)
    data = json.loads(resp.body)
    assert data["du"] == pytest.approx(2.0)

    bal_resp = await svc.get_balance("a")
    data = json.loads(bal_resp.body)
    assert data["ip"] == pytest.approx(2.0)
    assert data["du"] == pytest.approx(2.0)
