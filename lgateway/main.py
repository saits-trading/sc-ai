"""lgateway FastAPI app — mTLS termination, DPoP, Clerk-org routing, SSE streams."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lgateway.sse import router as sse_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sc-ai.lgateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start pub-sse-bridge if configured
    bridge_task = None
    if os.environ.get("BOT_SSE_URL") and os.environ.get("BOT_TENANT_ID"):
        from lbus.bridge import run_bridge
        bridge_task = asyncio.create_task(run_bridge())
        logger.info("pub-sse-bridge started")
    yield
    if bridge_task:
        bridge_task.cancel()
        try:
            await bridge_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="sc-ai lgateway",
    description="saitsCloud AI gateway — mTLS, DPoP, Clerk-org routing, SSE event streams",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(","),
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(sse_router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
