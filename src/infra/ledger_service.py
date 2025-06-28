from __future__ import annotations

import argparse
import json
import os
from typing import TYPE_CHECKING, Any

from .ledger import ledger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
else:  # pragma: no cover - optional dependency
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except Exception:
        from typing import Callable

        class FastAPI:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get(
                self, *args: object, **kwargs: object
            ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def dec(fn: Callable[..., Any]) -> Callable[..., Any]:
                    return fn

                return dec

            def post(
                self, *args: object, **kwargs: object
            ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def dec(fn: Callable[..., Any]) -> Callable[..., Any]:
                    return fn

                return dec

        class JSONResponse:
            def __init__(self, content: Any, *args: object, **kwargs: object) -> None:
                self.body = json.dumps(content).encode("utf-8")

        class BaseModel:  # pragma: no cover - minimal stub
            pass

        uvicorn = None  # type: ignore


app = FastAPI()


class SpendRequest(BaseModel):
    agent_id: str
    ip: float = 0.0
    du: float = 0.0
    reason: str = "spend"


class RewardRequest(BaseModel):
    agent_id: str
    ip: float = 0.0
    du: float = 0.0
    reason: str = "reward"


@app.get("/balance/{agent_id}")
async def get_balance(agent_id: str) -> JSONResponse:
    ip, du = await ledger.get_balance_async(agent_id)
    return JSONResponse({"ip": ip, "du": du})


@app.post("/spend")
async def spend(req: SpendRequest) -> JSONResponse:
    ip, du = await ledger.spend(req.agent_id, ip=req.ip, du=req.du, reason=req.reason)
    return JSONResponse({"ip": ip, "du": du})


@app.post("/reward")
async def reward(req: RewardRequest) -> JSONResponse:
    ip, du = await ledger.reward(req.agent_id, ip=req.ip, du=req.du, reason=req.reason)
    return JSONResponse({"ip": ip, "du": du})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ledger service.")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    return parser.parse_args(argv or [])


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.version:
        from src import __version__

        print(__version__)
        return
    host = os.getenv("LEDGER_HOST", "0.0.0.0")
    port = int(os.getenv("LEDGER_PORT", "8001"))
    if uvicorn is None:  # pragma: no cover - uvicorn missing
        raise RuntimeError("uvicorn is required to run the service")
    uvicorn.run(app, host=host, port=port)


__all__ = ["app", "get_balance", "reward", "spend"]
