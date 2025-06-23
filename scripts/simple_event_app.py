#!/usr/bin/env python3
"""Minimal FastAPI app that emits a test event via SSE."""

import asyncio
import sys
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
queue: asyncio.Queue[str | None] = asyncio.Queue()


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/stream/events")
async def stream_events() -> StreamingResponse:
    async def gen() -> AsyncGenerator[str, None]:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


async def main(port: int) -> None:
    async def push() -> None:
        await asyncio.sleep(0.1)
        await queue.put('{"event_type":"test","data":{"value":1}}')

    task = asyncio.create_task(push())
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    await server.serve()
    task.cancel()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    asyncio.run(main(port))
